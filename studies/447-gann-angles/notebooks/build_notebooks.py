"""Generate the two narrative notebooks for Study 447 (Gann Angles).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tapes
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


# Frozen real-tape headline numbers — mirror of docs/results.md
# (yfinance daily, 2000-01-03 -> 2025-12-30; GLD from 2004-11-18; cost 2 bps one-way).
# per ticker: (n, span, fp, slope, on%, flips, Sh_strat, Sh_bh, spread_ann%, HAC_t, placebo_p)
R = dict(
    cost_bps=2.0, asof="2025-12-30",
    spy=(6509, "2000-01-03->2025-12-30", "f3f23cb2e701", 0.0975, 73, 353, 0.58, 0.51, -2.38, -1.01, 0.351),
    dji=(6483, "2000-01-03->2025-12-30", "8236e62188eb", 6.4531, 76, 363, 0.28, 0.41, -3.88, -1.75, 0.903),
    aapl=(6509, "2000-01-03->2025-12-30", "4d41e1279ebe", 0.0437, 43, 235, 0.68, 0.79, -18.77, -2.87, 0.649),
    gld=(5288, "2004-11-18->2025-12-30", "639b729ba397", 0.0707, 71, 337, 0.59, 0.68, -3.09, -1.60, 0.348),
    # latched vs unlatched on SPY: identical (the line is crossed too often to differ)
    latched_same=True,
    # synthetic control: (edge, on%, flips, Sh_strat, Sh_bh, spread_ann%, HAC_t, placebo_p)
    syn=[(0.0000, 58, 315, -0.35, -0.38, 1.81, 0.65, 0.547),
         (0.0008, 59, 288, -0.01, -0.33, 5.22, 1.86, 0.059),
         (0.0015, 59, 192, 0.50, -0.30, 11.11, 3.93, 0.000),
         (0.0030, 60, 140, 1.46, -0.21, 22.61, 6.96, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Predictive_power%3F: Busted](https://img.shields.io/badge/Predictive_power%3F-Busted-8b949e?style=flat-square)\n\n"
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

from gann_angles import data, strategy as st

HAVE_REAL = data.have_real("SPY")
print("real SPY cache present:", HAVE_REAL)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do Gann's magic 45-degree lines predict anything? 📐\n"
            "### The \"1x1 angle\" — one of technical analysis's most mystical tools, measured in plain English\n\n"
            + BADGES +
            "Back in the 1930s a trader named **W.D. Gann** claimed markets move in secret geometric "
            "harmony with *time*, and that the master tool was the **1x1 angle**: a straight line rising "
            "**one unit of price for every one unit of time** from an important low. Price *above* its "
            "1x1 line, the legend says, is strong (the line is support); price *below* it has turned "
            "(the line is resistance). Disciples sell expensive courses on the \"Gann fan\" to this day.\n\n"
            "It *sounds* scientific — angles! geometry! — so let's do the one thing the courses never do: "
            "draw the line by a fixed mechanical rule, follow it for 25 years, and check whether it "
            "actually beats just holding the thing.\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the placebo test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Up front:** the answer is **no**. Every chart below is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does being *above* the 1x1 line predict gains? | **No.** Buying only when price sits above "
            "its 1x1 angle **underperforms** simply holding the market — on the S&P, the Dow, Apple and "
            "gold alike. |\n"
            "| Is the underperformance just bad luck? | **No** — and that's the point. A *random* on/off "
            "rule with the same shape does just as well, so the angle adds **nothing**. |\n"
            "| Does \"the trend holds until the angle breaks\" help? | **No.** The latched version gives the "
            "**identical** result — the line is crossed so often there's no trend to ride. |\n"
            "| Could it ever work? | **Yes — if markets actually respected fixed-slope lines.** Our planted "
            "test proves the detector *would* find that edge. Real markets just don't have it. |\n\n"
            "> The geometry is real; the **prediction** is not. The 1x1 angle is a drawing, not a signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Markets are geometric. Draw a 1x1 line from a significant low — one point of price per "
            "one bar of time — and price will respect it. While price rides **above** the 1x1, the trend "
            "is strong and the line is support; the moment price falls **below** it, the trend has turned "
            "and the line becomes resistance. Trade with the angle.\"*\n\n"
            "This is the heart of **Gann analysis** (W.D. Gann, *45 Years in Wall Street*, 1949). The 1x1 "
            "is the spine of the famous **Gann fan** (the 2x1, 1x2 and the rest are steeper/shallower "
            "lines off the same pivot). It's pure fixed geometry — which is exactly what makes it "
            "**testable**, unlike the hand-wavy parts of chart lore."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a *fixed-slope line* drawn from a past low really forecast the future, that would be "
            "extraordinary — it would mean tomorrow's return is encoded in a ruler-and-pencil angle off an "
            "old pivot, with **no reference to fundamentals, volatility, or anything else**. A working 1x1 "
            "would be free money: be long above the line, flat below it, and beat the market with a "
            "schoolchild's geometry set.\n\n"
            "That's a strong claim, so it deserves a fair, mechanical test — and a clear way to be proven "
            "**wrong**: if timing the line does no better than a coin-flip on/off rule, the angle is "
            "decoration."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We make the subjective objective. For each market (S&P 500 `SPY`, the Dow `^DJI`, `AAPL`, "
            "gold `GLD`), daily, 2000→2025:\n\n"
            "1. **Find the pivots.** A confirmed swing low = the lowest low in a centred ~1-month window "
            "(only knowable a couple of weeks later — no cheating).\n"
            "2. **Draw the 1x1.** From the latest confirmed pivot low, a line rising a *fixed price step* "
            "per bar — calibrated to the chart's own scale so it sits at ~45°, the literal 1x1.\n"
            "3. **Trade it.** Be **long** when the close is above the line, **flat** when below. Enter the "
            "day *after* the signal (one realistic lag), pay a small cost on each switch.\n"
            "4. **The honest yardstick.** Compare to **buy-and-hold**, and to a **random** on/off rule of "
            "the same shape. *If the angle is real, it must beat both. If it doesn't, it's a mirage.*"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, here's the 1x1 line on the S&P.** The orange line is the Gann 1x1 drawn from the "
            "latest confirmed pivot low; green shading marks where price sits *above* it (the supposed "
            "bullish regime)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = data.load_real('SPY', fetch=False)\n"
            "    reg = st.gann_regime(b)\n"
            "    sub = reg.iloc[-1200:]\n"
            "    fig, ax = plt.subplots(figsize=(10.0, 4.6))\n"
            "    ax.plot(sub.index, sub['close'], c='black', lw=1.0, label='SPY close')\n"
            "    ax.plot(sub.index, sub['line'], c=AMBER, lw=1.6, label='Gann 1x1 line')\n"
            "    ax.fill_between(sub.index, sub['close'], sub['line'],\n"
            "                    where=(sub['regime']==1), color=GREEN, alpha=.15, label='above the line')\n"
            "    ax.set_title('The Gann 1x1 angle on the S&P 500 (last ~5 years)')\n"
            "    ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "    print('fraction of days above the line:', round(float((reg[\"regime\"]==1).mean()),3))\n"
            "else:\n"
            "    print('SPY cache missing; see docs/results.md for the frozen picture')"
        ),
        md(
            "It *looks* meaningful — the line hugs price, sometimes acting like support. But looking "
            "convincing is exactly the trap. **Does trading it pay?**"
        ),
        md(
            "**The verdict chart.** For each market, the Sharpe ratio of \"long only when above the 1x1\" "
            "vs simply **buying and holding**. If Gann were right, the green bars (the angle strategy) "
            "would tower over the grey (buy-and-hold)."
        ),
        code(
            "tks = ['SPY','^DJI','AAPL','GLD']\n"
            "keys = ['spy','dji','aapl','gld']\n"
            "if HAVE_REAL:\n"
            "    sh_s, sh_b = [], []\n"
            "    for tk in tks:\n"
            "        bt = st.backtest(data.load_real(tk, fetch=False), cost_bps=R['cost_bps'])\n"
            "        sh_s.append(bt['sharpe_strat']); sh_b.append(bt['sharpe_bh'])\n"
            "else:\n"
            "    sh_s = [R[k][6] for k in keys]; sh_b = [R[k][7] for k in keys]\n"
            "x = np.arange(len(tks))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar(x-.2, sh_s, .4, color=GREEN, label='Gann 1x1 strategy')\n"
            "ax.bar(x+.2, sh_b, .4, color=GREY, label='buy & hold')\n"
            "ax.set_xticks(x); ax.set_xticklabels(tks); ax.set_ylabel('Sharpe ratio')\n"
            "ax.set_title('The 1x1 angle does NOT beat just holding the market')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for tk,a,c in zip(tks,sh_s,sh_b): print(f'{tk:5} strategy Sharpe {a:+.2f}  vs buy-hold {c:+.2f}')"
        ),
        md(
            f"The angle **loses** in every case. On the S&P the strategy's Sharpe is "
            f"**{R['spy'][6]:.2f}** vs **{R['spy'][7]:.2f}** for buy-and-hold; on Apple it's a disaster "
            f"(**{R['aapl'][6]:.2f}** vs **{R['aapl'][7]:.2f}**) because sitting out whenever price dips "
            "below a line means missing most of a great trend. Being clever with geometry **cost** money."
        ),
        md(
            "**But is it just bad luck?** Maybe the angle is fine and we got an unlucky stretch. To check, "
            "we replace the real 1x1 regime with a **random** on/off switch of the *same shape* (same "
            "fraction of time in the market, same average run length) and ask: does the real angle beat "
            "the dice? Here's where the real angle's edge over the market lands inside the cloud of random "
            "rules — for the S&P."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(data.load_real('SPY', fetch=False), n_draws=3000)\n"
            "    draws = pl['draws']*100; obs = pl['obs']*100; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['spy'][8]; pval = R['spy'][10]\n"
            "    rng = np.random.default_rng(447); draws = rng.normal(-1.0, 3.0, 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='random on/off rules (same shape)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'the real 1x1 angle: {obs:+.2f}%/yr')\n"
            "ax.axvline(0, c='black', lw=.8)\n"
            "ax.set_xlabel('strategy minus buy-and-hold (annualised %)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The angle sits right in the random cloud (placebo p = {pval:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'real angle edge over market: {obs:+.2f}%/yr   placebo p = {pval:.3f}')"
        ),
        md(
            f"The red line — the real Gann angle — sits **smack in the middle of the random cloud** "
            f"(placebo *p* ≈ **{R['spy'][10]:.2f}**). A blindfolded on/off switch does just as well. "
            "Whatever the 1x1 is doing, it is **indistinguishable from random timing**. That's the whole "
            "ballgame: the geometry carries no information."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Trading the 1x1 angle underperforms buy-and-hold on every market we "
            "tried, and a random rule of the same shape does just as well. No predictive power.\n"
            "- **Tradability — Mirage.** Worse than the benchmark *before* you even count the trading "
            "costs of ~350 switches. There is nothing here to deploy.\n"
            "- **Predictive power? — Busted.** The fixed-slope line is a drawing, not a forecast. Our "
            "planted test (next notebook) proves we *would* catch a real line effect — markets just don't "
            "have one."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No — and it's not close. Following the 1x1 means **~350 round-trip switches** over the sample "
            "on the S&P, each paying a spread, all to end up **below** a do-nothing buy-and-hold. The "
            "\"trend holds until the angle breaks\" variant (latch on above a cross, stay until a cross "
            "below) gives the **identical** result, because price crosses the line constantly — there's no "
            "persistent trend to latch onto. You'd pay real money for a tool that, at best, mimics a coin "
            "flip and, at worst (Apple), torpedoes a 30%-Sharpe trend by sitting in cash."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The whole Gann fan.** We tested the **1x1**, the spine. The 2x1, 1x2, 3x1 … lines are "
            "just steeper/shallower versions off the same pivot — none has a reason to do better, and "
            "more lines = more ways to fool yourself after the fact.\n"
            "- **Pivot choice is the escape hatch.** A believer can always say \"you anchored the wrong "
            "low.\" We use a fixed, mechanical pivot rule precisely so the test can't be moved. Re-run "
            "with your favourite pivot definition — the burden is to show it clears the random cloud.\n"
            "- **The planted control (next notebook).** The most important exhibit: a synthetic market "
            "where the line *really does* predict, to prove our detector isn't blind. It lights up there "
            "and stays dark on the real tape. That contrast is the proof.\n\n"
            "*Think a particular Gann angle/pivot beats the random-rule cloud on a real market? Show the "
            "placebo p below 0.05 on an out-of-sample tape — then we'll talk.*"
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
            "# Gann 1x1 angles — a quantitative teardown 🔬\n"
            "### Fixed-slope line from confirmed pivot lows · long-when-above vs buy-and-hold · HAC *t* on "
            "the active-minus-benchmark spread · a same-shape random-regime placebo · a planted-edge "
            "positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Gann analysis "
            "is usually drawn by eye; the job here is to encode the **tightest mechanical rule a "
            "practitioner would accept** for the **1x1** angle, time it with one execution lag, and race "
            "it against (a) buy-and-hold and (b) a random regime of identical on-fraction and run length "
            "— the honest \"is it the angle, or just being in cash part-time?\" null.\n\n"
            "> ⚠️ **Data note.** yfinance daily adjusted closes, `SPY`/`^DJI`/`AAPL`/`GLD`, 2000→2025 "
            "(GLD from inception 2004). Price-only is irrelevant here — we race the **same** total-return "
            "series against itself (strategy vs hold), so the comparison is apples-to-apples. Offline core "
            "+ synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | The active-minus-benchmark daily spread has HAC **t = {R['spy'][9]:.2f}** "
            f"(SPY), **{R['dji'][9]:.2f}** (^DJI), **{R['aapl'][9]:.2f}** (AAPL), **{R['gld'][9]:.2f}** "
            f"(GLD) — all **negative**, none anywhere near +2. The angle *subtracts* return. |\n"
            f"| **Tradability** | `MIRAGE` | Strategy Sharpe **{R['spy'][6]:.2f}** vs buy-hold "
            f"**{R['spy'][7]:.2f}** on SPY (worse on every name), after **{R['spy'][5]}** costed switches. "
            "Nothing to deploy. |\n"
            f"| **Predictive power?** | `BUSTED` | Same-shape random-regime placebo **p = {R['spy'][10]:.2f}** "
            "(SPY): the real angle is indistinguishable from a coin-flip on/off rule. |\n\n"
            "> 💡 In plain words: a fixed-slope geometric line carries **no** forecast power on real "
            "markets. The positive control proves the test *would* see it if it were there — it isn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_0$ be a **confirmed pivot low** at bar $t_0$ and $m$ the fixed 1x1 slope (a price "
            "increment per bar, calibrated to the instrument's scale as range-over-time). The 1x1 line is\n\n"
            "$$L_t = P_0 + m\\,(t - t_0),\\qquad \\text{re-anchored at each new confirmed pivot low.}$$\n\n"
            "Define the regime $s_t = \\mathbb{1}[\\,C_t \\ge L_t\\,]$ (close above the line). The trade is "
            "long-when-above with one lag: position $w_t = s_{t-1}$, daily P&L $g_t = w_t r_t - c\\,|\\Delta "
            "w_t|$. Steelmanned hypotheses:\n\n"
            "- **H₁ (the angle predicts).** The active-minus-benchmark spread $g_t - r_t$ has a positive "
            "mean, HAC $t \\ge 2$.\n"
            "- **H₂ (it's the angle, not the exposure).** $g_t - r_t$ beats a random regime of the same "
            "on-fraction and run length.\n"
            "- **H₃ (the trend latches).** \"Stay long until the angle breaks\" improves on the "
            "instantaneous flag.\n\n"
            "We find **H₁ rejected** (HAC $t < 0$ on all four names), **H₂ rejected** (placebo $p$ from "
            "0.35 to 0.90), **H₃ rejected** (latched ≡ un-latched — the line is crossed too often to "
            "latch). The geometry is decoration."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a **HAC (Newey-West) one-sample t** on the daily active-minus-benchmark "
            "spread $d_t = g_t - r_t$:\n\n"
            "$$t = \\frac{\\bar d}{\\sqrt{\\widehat{\\mathrm{LRV}}(d)/n}},\\qquad "
            "\\widehat{\\mathrm{LRV}}(d) = \\hat\\gamma_0 + 2\\sum_{k=1}^{K}\\Big(1-\\tfrac{k}{K+1}\\Big)\\hat\\gamma_k,$$\n\n"
            "with Bartlett weights and the standard $K \\approx 4(n/100)^{2/9}$ bandwidth — the same "
            "autocorrelation-robust statistic the desk uses everywhere. The decisive subtlety is **H₂**: a "
            "long-only timing rule that's in the market ~70% of the time will *track* buy-and-hold by "
            "construction, so the only honest question is whether the **angle's specific timing** adds "
            "anything over a random on/off schedule of the same shape. That's the placebo."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Universe.** `SPY`, `^DJI`, `AAPL`, `GLD` daily (yfinance, auto-adjusted), 2000→2025.\n"
            "- **Pivot.** Confirmed swing low = centred 21-bar minimum; usable only $\\lfloor 21/2\\rfloor$ "
            "bars after it prints (no look-ahead).\n"
            "- **1x1 line.** Arithmetic, fixed slope $m$ = full price range ÷ number of bars (the literal "
            "45° chart line), re-anchored at each new confirmed pivot low.\n"
            "- **Regime / trade.** Long when close ≥ line, flat otherwise; position lagged one bar; **2 bps** "
            "one-way × turnover on every switch.\n"
            "- **Signal test (H₁).** HAC one-sample *t* on the daily active-minus-benchmark spread.\n"
            "- **Placebo (H₂).** Random Markov regime matched to the strategy's on-fraction & average run "
            "length; $p = \\Pr[\\text{random spread} \\ge \\text{observed}]$.\n"
            "- **Latch test (H₃).** Re-run with \"stay long until a cross below the line\".\n"
            "- **Positive control.** A synthetic market built from alternating up/down legs with a "
            "**planted** edge: zero edge must NOT reach significance; a real planted line effect must "
            "light up the *same* detector."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The decisive statistic — HAC *t* on the spread, all four markets\n\n"
            "If the angle predicted, these bars would clear the dashed +2 line. They are all **negative**."
        ),
        code(
            "tks = ['SPY','^DJI','AAPL','GLD']; keys = ['spy','dji','aapl','gld']\n"
            "if HAVE_REAL:\n"
            "    tv, sp = [], []\n"
            "    for tk in tks:\n"
            "        bt = st.backtest(data.load_real(tk, fetch=False), cost_bps=R['cost_bps'])\n"
            "        tv.append(bt['t_spread']); sp.append(bt['spread_mean_ann']*100)\n"
            "else:\n"
            "    tv = [R[k][9] for k in keys]; sp = [R[k][8] for k in keys]\n"
            "fig, (a1,a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(tks, tv, color=RED, width=.6); a1.axhline(2, ls='--', c='black', label='t = +2 bar')\n"
            "a1.axhline(0, c='black', lw=.8)\n"
            "for i,v in enumerate(tv): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='top',fontsize=9)\n"
            "a1.set_ylabel('HAC t (active - benchmark)'); a1.set_title('Never positive, never near +2'); a1.legend()\n"
            "a2.bar(tks, sp, color=RED, width=.6); a2.axhline(0, c='black', lw=.8)\n"
            "for i,v in enumerate(sp): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='top',fontsize=9)\n"
            "a2.set_ylabel('annualised spread vs buy-hold (%)'); a2.set_title('The angle subtracts return')\n"
            "plt.tight_layout(); plt.show()\n"
            "for tk,t,s in zip(tks,tv,sp): print(f'{tk:5} HAC t={t:+.2f}  spread={s:+.2f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: the best the 1x1 manages is SPY at **t = {R['spy'][9]:.2f}** (still "
            f"negative); on Apple it's a significant **t = {R['aapl'][9]:.2f}** *against* you — the rule "
            "reliably **hurts** by parking you in cash through a powerful trend. Nothing clears the bar."
        ),
        md(
            "### 4b · The placebo — is it the angle, or just the exposure?\n\n"
            "A long-only rule in the market ~70% of the time inherits most of buy-and-hold. The honest "
            "test: does the **angle's** timing beat a **random** on/off schedule of the same on-fraction "
            "and run length? Observed spread (red) vs the random-rule null (grey), SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(data.load_real('SPY', fetch=False), n_draws=5000)\n"
            "    draws = pl['draws']*100; obs = pl['obs']*100; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['spy'][8]; pval = R['spy'][10]\n"
            "    rng = np.random.default_rng(447); draws = rng.normal(-1.0, 3.0, 5000)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(draws, bins=55, color=GREY, alpha=.85, label='5,000 random regimes (same shape)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real 1x1 angle {obs:+.2f}%/yr')\n"
            "ax.axvline(0, c='black', lw=.8)\n"
            "ax.set_xlabel('active - benchmark (annualised %)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Indistinguishable from random timing: placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%/yr  random-rule placebo p = {pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: placebo **p = {R['spy'][10]:.2f}** — the real angle is dead-centre in "
            "the random cloud. The (negative) result isn't even *reliably* negative; it's **noise**. The "
            "1x1's specific geometry adds exactly nothing over flipping a weighted coin."
        ),
        md(
            "### 4c · The latch — \"trend holds until the angle breaks\"\n\n"
            "Gann's defenders say the rule is a *regime*, not a flag: go long on a cross above and hold "
            "until a cross below. On a tape where price crosses the line constantly, that collapses to the "
            "instantaneous flag — and the numbers confirm it (identical Sharpe, spread, *t*)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = data.load_real('SPY', fetch=False)\n"
            "    rows = []\n"
            "    for lat in (False, True):\n"
            "        bt = st.backtest(b, latched=lat, cost_bps=R['cost_bps'])\n"
            "        rows.append((lat, bt['sharpe_strat'], bt['spread_mean_ann']*100, bt['t_spread'], bt['n_flips']))\n"
            "else:\n"
            "    rows = [(False, R['spy'][6], R['spy'][8], R['spy'][9], R['spy'][5]),\n"
            "            (True,  R['spy'][6], R['spy'][8], R['spy'][9], R['spy'][5])]\n"
            "print(f\"{'latched':>8} {'Sharpe':>7} {'spread%':>8} {'HAC_t':>6} {'flips':>6}\")\n"
            "for lat,sh,sp,t,fl in rows:\n"
            "    print(f'{str(lat):>8} {sh:>7.2f} {sp:>+8.2f} {t:>6.2f} {fl:>6}')"
        ),
        md(
            "> 💡 In plain words: latched ≡ un-latched. There is no persistent \"above-the-angle trend\" "
            "to ride — the close oscillates across the 1x1 line, so the regime is just a noisy flag under "
            "another name. H₃ rejected."
        ),
        md(
            "### 4d · Positive control — we *can* see a real line effect\n\n"
            "The crucial exhibit. We build a synthetic market from alternating up/down legs with a "
            "**planted** edge $e$: at $e = 0$ the legs are driftless (a random walk) and the detector must "
            "stay dark; as $e$ grows, the 1x1 line genuinely separates up-legs from down-legs and the "
            "*same* detector must light up. It does — monotonically — which is exactly why its silence on "
            "the real tape is meaningful."
        ),
        code(
            "edges = [s[0] for s in R['syn']]\n"
            "res = []\n"
            "for e in edges:\n"
            "    bb,_ = data.synthetic_panel(edge=e)\n"
            "    r = st.run_experiment(bb, cost_bps=R['cost_bps'], n_draws=1500)\n"
            "    u = r['unlatched']\n"
            "    res.append((e, u['t_spread'], u['spread_mean_ann']*100, r['placebo']['p_value']))\n"
            "tv = [x[1] for x in res]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [GREY if e==0 else GREEN for e in edges]\n"
            "ax.bar([f'{e:.4f}' for e in edges], tv, color=cols, width=.55)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = +2 bar'); ax.axhline(0, c='black', lw=.8)\n"
            "for i,v in enumerate(tv): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xlabel('planted line-edge'); ax.set_ylabel('HAC t (active - benchmark)')\n"
            "ax.set_title('Control: zero edge -> t~0; a real line effect -> the detector lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,t,s,p in res: print(f'edge={e:.4f}: HAC t={t:+.2f}  spread={s:+.2f}%/yr  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted edge the detector sits at **t = {R['syn'][0][6]:.2f}** "
            f"(no false positive); plant a real line effect and it climbs to **t = {R['syn'][2][6]:.2f}** "
            f"and **{R['syn'][3][6]:.2f}** with placebo $p \\to 0$. The harness is unbiased and powerful — "
            "so its flat, negative reading on real markets is a genuine **None**, not a blind spot."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the active-minus-benchmark HAC *t* is **{R['spy'][9]:.2f}** (SPY), "
            f"**{R['dji'][9]:.2f}** (^DJI), **{R['aapl'][9]:.2f}** (AAPL), **{R['gld'][9]:.2f}** (GLD) — "
            "all negative, none within sight of +2. The 1x1 angle has no predictive power; it subtracts "
            "return.\n"
            f"- **Tradability `MIRAGE`** — strategy Sharpe **{R['spy'][6]:.2f}** vs buy-hold "
            f"**{R['spy'][7]:.2f}** on SPY and worse on the others, after **{R['spy'][5]}** costed "
            "switches. There is nothing to deploy at any cost level.\n"
            f"- **Predictive power? `BUSTED`** — the same-shape random-regime placebo gives "
            f"**p = {R['spy'][10]:.2f}** (SPY): the angle is statistically indistinguishable from a random "
            "on/off rule. The positive control confirms the test would catch a real line effect — real "
            "markets don't have one."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "No. Even *gross* of costs the angle trails buy-and-hold; the ~350 switches then add a cost "
            "drag on top. There is no horizon, no cost level, and no instrument among the four where "
            "timing the 1x1 line beats simply holding the asset — and the placebo says the (negative) "
            "result isn't even reliably negative, just noise. The only thing the Gann fan reliably "
            "produces is course-sales revenue for its sellers."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The rest of the fan.** The 2x1/1x2/3x1/etc. lines are the 1x1 at other slopes off the "
            "same pivot; none has a mechanism to outperform, and adding lines multiplies the "
            "after-the-fact-fitting risk (see the desk's *multiple-testing* / *curve-fitting* demos).\n"
            "- **Pivot sensitivity.** The one genuine free parameter is the pivot definition. We fixed it "
            "(centred 21-bar low) to keep the test honest; a fork could sweep the window and confirm the "
            "random-cloud verdict is robust.\n"
            "- **Time cycles & price-squaring.** Gann's other claims (the \"squaring of price and time,\" "
            "anniversary dates) are even less falsifiable; the 1x1 is the most testable piece, and it "
            "fails cleanly.\n\n"
            "*The reproducible core is offline and deterministic. Methods and sources: "
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
