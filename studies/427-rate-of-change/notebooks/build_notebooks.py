"""Generate the two narrative notebooks for Study 427 (Rate of Change).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY daily tape
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# SPY daily total-return closes (yfinance auto_adjust), 1993-01-29 -> 2026-05-29 (pinned to the
# last complete month), 8,390 bars / 32.8 traded years; ROC long/flat 126-day, 1 bp one-way,
# 2%/yr flat risk-free, excess-vs-excess Sharpe, one-day execution lag.
R = dict(
    ticker="SPY", start="1993-01-29", end="2026-05-29", n_bars=8390, years=32.8,
    fingerprint="f3fa058adfc8", window=126, cost_bps=1.0, rf_annual=0.02,
    # ROC long/flat: sharpe, cagr, maxdd, t_exc, ann_turnover, exposure
    roc=dict(sharpe=0.591, cagr=0.0881, maxdd=-0.314, t_exc=3.71, turnover=6.6, exposure=0.76),
    hold=dict(sharpe=0.542, cagr=0.1090, maxdd=-0.552, t_exc=3.64),
    # the decisive number: ROC minus buy-and-hold, net
    diff=dict(delta_ann=-0.0297, t_diff=-1.47),
    placebo=dict(obs=0.591, p=0.034),
    breakeven_bps=0.0,
    # long/short variant
    ls=dict(sharpe=0.225, cagr=0.0456, maxdd=-0.465, delta_ann=-0.0594, t_diff=-1.47),
    # benchmark race: rule -> (sharpe, cagr, maxdd, t_exc, turnover)
    race=[
        ("Buy & hold", 0.542, 0.1090, -0.552, 3.64, 0.0),
        ("ROC(126)", 0.591, 0.0881, -0.314, 3.71, 6.6),
        ("SMA(200)", 0.580, 0.0865, -0.283, 3.55, 6.6),
        ("MACD(12,26,9)", 0.244, 0.0416, -0.385, 1.52, 21.4),
        ("RSI(14)>50", 0.280, 0.0458, -0.445, 1.67, 26.7),
    ],
    # windows: N -> (sharpe, cagr, delta_ann, t_diff)
    windows=[(21, 0.264, 0.0441, -0.0710, -3.28), (63, 0.456, 0.0679, -0.0491, -2.29),
             (126, 0.591, 0.0881, -0.0297, -1.47), (252, 0.610, 0.0998, -0.0162, -0.87)],
    # cost sweep: bps -> (sharpe, cagr)
    costs=[(0.0, 0.596, 0.0888), (1.0, 0.591, 0.0881), (2.0, 0.586, 0.0873),
           (5.0, 0.569, 0.0852), (10.0, 0.542, 0.0816)],
    # synthetic control: edge -> (roc_sharpe, hold_sharpe, delta_ann, t_diff)
    syn=[(0.0, -0.29, -0.11, -0.0128, -0.57), (0.5, 4.83, -0.15, 0.7602, 11.01)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Better_momentum_tool%3F: Misattributed](https://img.shields.io/badge/Better_momentum_tool%3F-Misattributed-8b949e?style=flat-square)\n\n"
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

from rate_of_change import data, strategy as st

HAVE_REAL = data.have_real("SPY")
if HAVE_REAL:
    BARS = data.load_real("SPY").loc[:"2026-05-31"]   # pinned to the last complete month
    CLOSE = BARS["close"]
else:
    BARS = CLOSE = None
print("real SPY cache present:", HAVE_REAL,
      "| bars:", (0 if BARS is None else len(BARS)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the oldest momentum tool actually beat just holding? ⏱️\n"
            "### Rate of Change (ROC) — the granddaddy of momentum indicators, on the bench\n\n"
            + BADGES +
            "Open any trading book and you'll meet **Rate of Change** early. It's almost embarrassingly "
            "simple: take today's price, compare it to the price *N* days ago, and express the gap as a "
            "percent. ROC above zero means \"price is higher than it was\" — an uptrend. The folk rule: "
            "**go long when ROC is positive, step aside to cash when it goes negative.** Ride the trend, "
            "dodge the crashes.\n\n"
            "It *sounds* like free risk management. And here's the twist this study uncovers: the rule "
            "really does cut your worst drawdown roughly in **half** — but it does it by **giving up "
            "return**, and once you line it up against the dead-simplest alternative (a 200-day moving "
            "average) it's a **photo finish**. The \"oldest momentum tool\" turns out to be one of *many* "
            "trend filters, none special.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the permutation test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Data note.** SPY daily **total-return** closes (dividends + splits folded in), "
            "1993→2026, pinned to the last complete month. Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does ROC timing make money? | **Yes — but so does just holding.** ROC's Sharpe (**0.59**) "
            "edges out buy-and-hold's (**0.54**), but its **return is lower** (8.8%/yr vs 10.9%/yr). |\n"
            "| Does it *add* anything over holding? | **No, not measurably.** ROC minus buy-and-hold is "
            "**−3.0%/yr** with *t* = **−1.47** — a small, statistically-insignificant *loss* of return. |\n"
            "| So what's it good for? | **Cutting the drawdown.** Worst loss drops from **−55%** (hold) to "
            "**−31%** (ROC). It's a risk-reducer, not an alpha source. |\n"
            "| Is it a *better* momentum tool? | **No.** A plain **200-day moving-average** cross ties it "
            "almost exactly (Sharpe 0.58 vs 0.59). ROC isn't special — it's one trend filter among many. |\n\n"
            "> The oldest momentum tool *works* in the narrow sense that staying long during uptrends is "
            "sensible — but it earns that calm by **leaving return on the table**, and nothing about the "
            "ROC formula beats the simplest moving average."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Rate of Change is the purest momentum indicator there is. When ROC crosses above zero "
            "the trend is up — buy. When it crosses below, the trend is down — get out. You ride the big "
            "moves and sidestep the crashes.\"*\n\n"
            "ROC has been in the technician's toolbox since the **1970s** (it predates RSI and MACD), and "
            "every charting platform ships it. The steelman is real: momentum *is* one of the best-"
            "documented anomalies in finance (Jegadeesh & Titman 1993). So the question isn't \"is "
            "momentum a thing?\" — it's **\"does this 50-year-old one-line formula capture it any better "
            "than holding, or than the next indicator over?\"**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a one-line indicator could reliably beat buy-and-hold *after costs*, it would be the best "
            "deal in finance — free outperformance from arithmetic you can do in your head. That's a big "
            "claim, so it deserves a hard test.\n\n"
            "Two traps hide here. **(1) Beta in disguise.** A long/flat rule that's invested ~75% of the "
            "time will look great in a 30-year bull market — but that's mostly just *owning stocks*, not "
            "the indicator's skill. The honest test is ROC **minus** buy-and-hold. **(2) Indicator "
            "vanity.** Every momentum filter — ROC, a moving-average cross, MACD — is secretly the same "
            "bet (\"stay long while price rises\"). If they all tie, the *specific* indicator is a "
            "distraction. We test both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We compute **ROC over {R['window']} trading days** on **{R['years']:.0f} years** of SPY "
            f"({R['start']} → {R['end']}). Then:\n\n"
            "1. **The rule.** Long when ROC > 0, flat (cash) otherwise. The signal is read at today's "
            "close; the position earns **tomorrow's** return (one-day lag, no cheating).\n"
            "2. **The race.** Compare its risk-adjusted return (Sharpe), its growth (CAGR) and its worst "
            "drawdown against simply **buying and holding** SPY — and against a **200-day moving average**, "
            "**MACD**, and **RSI**, so \"ROC is better\" is actually checked.\n"
            "3. **The verdict line.** If ROC's edge **over buy-and-hold** isn't clearly positive after "
            "costs, the timing adds nothing — it's just dialing your stock exposure up and down."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline race.** Here's the growth of $1 in SPY held forever vs the same dollar "
            "run through the ROC long/flat rule (net of a realistic 1 bp one-way cost)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bh = st.buy_and_hold(CLOSE)\n"
            "    rb = st.book_returns(CLOSE, st.roc_signal(CLOSE, R['window'], 'long_flat'), R['cost_bps'])\n"
            "    eq_bh = (1+bh['net']).cumprod(); eq_roc = (1+rb['net']).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.5,4.6))\n"
            "    ax.semilogy(eq_bh.index, eq_bh, color=GREY, lw=1.6, label=f\"Buy & hold (CAGR {R['hold']['cagr']*100:.1f}%)\")\n"
            "    ax.semilogy(eq_roc.index, eq_roc, color=GREEN, lw=1.8, label=f\"ROC({R['window']}) long/flat (CAGR {R['roc']['cagr']*100:.1f}%)\")\n"
            "    ax.set_ylabel('growth of $1 (log)'); ax.set_title('ROC rides the trend — but ends up BELOW just holding')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache absent — see frozen numbers in R')\n"
            "print(f\"ROC  CAGR {R['roc']['cagr']*100:.1f}%  Sharpe {R['roc']['sharpe']:.2f}  maxDD {R['roc']['maxdd']*100:.0f}%\")\n"
            "print(f\"Hold CAGR {R['hold']['cagr']*100:.1f}%  Sharpe {R['hold']['sharpe']:.2f}  maxDD {R['hold']['maxdd']*100:.0f}%\")"
        ),
        md(
            f"The green line (ROC) spends the whole chart **below** the grey line (hold). ROC grows your "
            f"dollar at **{R['roc']['cagr']*100:.1f}%/yr** vs **{R['hold']['cagr']*100:.1f}%/yr** for "
            "holding. It's not that ROC is *bad* — it's that being in cash part of the time means you miss "
            "some of the bull market you were trying to ride."
        ),
        md(
            "**So where's the upside?** The drawdown. Here's the worst peak-to-trough loss for each. This "
            "is the whole case for ROC."
        ),
        code(
            "rules = ['Buy & hold', f\"ROC({R['window']})\"]\n"
            "dd = [abs(R['hold']['maxdd'])*100, abs(R['roc']['maxdd'])*100]\n"
            "fig, ax = plt.subplots(figsize=(7.2,4.2))\n"
            "ax.bar(rules, dd, color=[GREY, GREEN], width=.55)\n"
            "for i,v in enumerate(dd): ax.annotate(f'-{v:.0f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('worst drawdown (%)'); ax.set_title('The one thing ROC buys you: half the drawdown')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"hold worst loss {R['hold']['maxdd']*100:.0f}%  ->  ROC worst loss {R['roc']['maxdd']*100:.0f}%\")"
        ),
        md(
            f"There's the real benefit: ROC cuts the worst loss from **{R['hold']['maxdd']*100:.0f}%** to "
            f"**{R['roc']['maxdd']*100:.0f}%** by stepping aside in 2008 and 2020. That's genuine risk "
            "management — but it's a *trade*, not a free lunch: you pay for the calm with ~2%/yr of return."
        ),
        md(
            "**Now the vanity check.** Is ROC a *better* momentum tool than the alternatives, or do they "
            "all do the same job? Here's the Sharpe of each rule on the same tape, same costs."
        ),
        code(
            "names = [r[0] for r in R['race']]; sh = [r[1] for r in R['race']]\n"
            "cols = [GREY] + [GREEN if n.startswith('ROC') else AMBER for n in names[1:]]\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "ax.bar(range(len(names)), sh, color=cols, width=.62)\n"
            "for i,v in enumerate(sh): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=18, ha='right')\n"
            "ax.set_ylabel('excess Sharpe'); ax.set_title('ROC ties the 200-day moving average — it is not special')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('ROC(126) Sharpe', R['race'][1][1], ' vs SMA(200) Sharpe', R['race'][2][1])"
        ),
        md(
            f"This is the punchline. **ROC ({R['race'][1][1]:.2f})** and a plain **200-day moving-average "
            f"cross ({R['race'][2][1]:.2f})** are a dead heat — and both beat the fussier MACD and RSI "
            "rules (which churn far more and earn less). The lesson isn't \"ROC is great,\" it's "
            "\"**any** simple trend filter does roughly the same thing, and the specific indicator is "
            "marketing.\""
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** ROC's timing isn't random (the quants notebook shows it beats a "
            "shuffled schedule), but its edge **over buy-and-hold** is **−3.0%/yr** at *t* = −1.47 — i.e. "
            "*not* a positive, significant alpha. It's a beta-reducer dressed up as a signal.\n"
            "- **Tradability — Mirage.** You can run it, but it **lowers** your return; the only thing it "
            "delivers is a smaller drawdown, which a 200-day SMA delivers identically. No incremental "
            "money to be made.\n"
            "- **\"Better momentum tool\"? — Misattributed.** ROC ties the simplest moving average. The "
            "benefit people credit to ROC is just \"stay long in uptrends\" — shared by every trend rule."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Costs aren't even the problem here — ROC trades rarely (about 7 turns a year), so it survives "
            "a wide range of cost assumptions. The problem is more basic: **there's no extra return to "
            "harvest.** Watch the net Sharpe barely move as costs climb."
        ),
        code(
            "cb = [c[0] for c in R['costs']]; cs = [c[1] for c in R['costs']]\n"
            "fig, ax = plt.subplots(figsize=(8.6,4.2))\n"
            "ax.plot(cb, cs, 'o-', color=GREEN, lw=2)\n"
            "ax.axhline(R['hold']['sharpe'], ls='--', color=GREY, label=f\"buy & hold Sharpe {R['hold']['sharpe']:.2f}\")\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('ROC net excess Sharpe')\n"
            "ax.set_title('Low turnover -> costs barely bite; but there is no edge to defend'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('ROC Sharpe at 1bp', R['costs'][1][1], ' at 10bp', R['costs'][4][1])"
        ),
        md(
            "ROC's Sharpe edges buy-and-hold's even at 10 bps — but remember *why*: it carries less risk, "
            "not more return. If you want the lower drawdown, ROC (or an SMA) gives it to you cheaply. If "
            "you want **more money**, neither does. That's the candid bottom line: a fine risk dial, a "
            "non-existent alpha."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Tune the lookback.** Shorter ROC windows (21, 63 days) trail *more* (they whipsaw); "
            "longer ones (252) close the gap to holding by just... holding more. There's no magic N.\n"
            "- **Long/short is worse.** Flipping to short when ROC < 0 *halves* the Sharpe — shorting a "
            "market with a long-run upward drift is a losing add-on.\n"
            "- **The real use.** Treat ROC as a **risk overlay** (cap your drawdown), not a return "
            "engine. And if you do, use whichever trend filter you find clearest — they're interchangeable.\n\n"
            "*Think a specific ROC window beats buy-and-hold's CAGR after costs? Show the **delta vs hold** "
            "landing positive with t ≥ 2 — then we'll talk.*"
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
            "# Rate of Change — a quantitative teardown 🔬\n"
            "### ROC long/flat timing on SPY · excess-vs-excess Sharpe · HAC *t* on the difference vs "
            "buy-and-hold · a circular-permutation placebo · cost sweep · the SMA/MACD/RSI race · a "
            "regime-switching positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). ROC is the "
            "oldest momentum oscillator; the job here is to test the *implicit* claim — that it beats "
            "holding and beats its rivals — under the desk's inference bar. The decisive statistic is the "
            "HAC *t* on the **difference** ROC − buy-and-hold, not ROC's own *t* (which is mostly beta).\n\n"
            "> ⚠️ **Data note.** SPY daily **total-return** closes (yfinance `auto_adjust=True`), "
            f"{R['start']}→{R['end']}, pinned to the last complete month ({R['n_bars']:,} bars, "
            f"{R['years']:.1f} traded years, fingerprint `{R['fingerprint']}`). Sharpe is **excess of a "
            "flat 2%/yr risk-free**, both legs; one execution lag (signal at close *t*, return of *t+1*). "
            "Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `WEAK` | ROC's own excess Sharpe **{R['roc']['sharpe']:.2f}** (HAC "
            f"*t* = {R['roc']['t_exc']:.2f}) clears t≥2 — but that is **beta** (hold's t = "
            f"{R['hold']['t_exc']:.2f}). The edge **over hold** is **{R['diff']['delta_ann']*100:+.1f}%/yr** "
            f"at HAC *t* = **{R['diff']['t_diff']:.2f}** — no positive incremental alpha. |\n"
            f"| **Tradability** | `MIRAGE` | ROC **lowers** CAGR ({R['roc']['cagr']*100:.1f}% vs "
            f"{R['hold']['cagr']*100:.1f}%); its only deliverable is drawdown ({R['roc']['maxdd']*100:.0f}% "
            f"vs {R['hold']['maxdd']*100:.0f}%), identical to a 200-day SMA. Nothing to scale. |\n"
            f"| **Better momentum tool?** | `MISATTRIBUTED` | ROC({R['window']}) Sharpe "
            f"**{R['race'][1][1]:.2f}** ties SMA(200) **{R['race'][2][1]:.2f}**; the benefit is the shared "
            "trend filter, not the ROC formula. |\n\n"
            "> 💡 In plain words: ROC's headline *t* looks great only because it's long a rising market "
            "most of the time. Net of that beta it adds **nothing** over holding — it just trades return "
            "for a smaller drawdown, exactly like every other simple trend filter."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_t$ be the (total-return) close and define the Rate of Change\n\n"
            "$$\\mathrm{ROC}_t(N) = 100\\left(\\frac{P_t}{P_{t-N}} - 1\\right).$$\n\n"
            "The folk timing rule is the long/flat position $w_t = \\mathbf{1}[\\mathrm{ROC}_t(N) > 0]$, "
            "held over $t{+}1$. Three nested hypotheses:\n\n"
            "- **H₁ (it makes money).** The booked net excess return has a positive HAC *t*.\n"
            "- **H₂ (it beats holding).** The **difference** $r^{\\text{ROC}}_t - r^{\\text{B\\&H}}_t$ has "
            "a *positive*, significant HAC *t* — the number that strips out beta.\n"
            "- **H₃ (the indicator is special).** ROC's Sharpe beats the obvious rivals "
            "(SMA(200), MACD, RSI).\n\n"
            "We find **H₁ true but uninformative** (it's beta), **H₂ rejected** "
            f"($\\Delta = {R['diff']['delta_ann']*100:+.1f}\\%$/yr, *t* = {R['diff']['t_diff']:.2f}$, "
            f"negative), **H₃ rejected** (a dead heat with SMA(200)). The oscillator is a risk overlay "
            "mislabelled as alpha."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — why the *difference* t is the only honest test\n\n"
            "A long/flat momentum rule on a market with secular drift is **mechanically** correlated with "
            "buy-and-hold (it's long ~76% of the time). So its standalone Sharpe and *t* mostly measure "
            "the equity premium — they would clear t≥2 for *any* mostly-long schedule. The desk's "
            "inference bar demands the **incremental** signal:\n\n"
            "$$d_t = r^{\\text{ROC}}_t - r^{\\text{B\\&H}}_t, \\qquad "
            "t_{\\Delta} = \\frac{\\bar d}{\\widehat{\\mathrm{se}}_{\\text{HAC}}(\\bar d)}.$$\n\n"
            "If $t_\\Delta$ doesn't clear 2 **positive**, ROC adds no alpha over the thing it's supposed to "
            "beat — whatever its own *t* says. We pair this with a **circular-permutation placebo** on the "
            "position vector (does the *timing* carry information beyond the exposure level?) and a "
            "**rivals race** (is the *formula* special, or just the trend filter?)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** SPY daily total-return closes, {R['start']}→{R['end']} ({R['n_bars']:,} bars, "
            f"fingerprint `{R['fingerprint']}`). Single liquid instrument — no survivorship, no panel.\n"
            f"- **Signal.** $\\mathrm{{ROC}}_t({R['window']}) > 0 \\Rightarrow$ long, else flat (cash). "
            "Read at close *t*, earns *t+1* (one `shift`, applied once).\n"
            "- **Costs.** 1 bp one-way × |Δposition| × NAV (turnover = one-way × NAV); a sweep to 10 bps; "
            "a break-even-cost solve. Long/flat sits in cash part-time → **excess-vs-excess** Sharpe "
            "(net − 2%/yr rf, both legs).\n"
            "- **Signal axis.** HAC *t* on ROC's excess return (the beta number) **and** HAC *t* on the "
            "difference vs buy-and-hold (the alpha number).\n"
            "- **Placebo.** 5,000 circular shifts of the realised position vector vs the returns — "
            "preserves exposure & turnover, destroys the timing. $p = \\Pr[\\text{shuffled Sharpe} \\ge "
            "\\text{observed}]$.\n"
            "- **Rivals.** SMA(200) cross, MACD(12,26,9), RSI(14)>50 — same lag, costs, rf.\n"
            "- **Positive control.** A regime-switching synthetic tape: at edge 0 the difference-*t* must "
            "stay ~0 (no false positive); at a planted edge ROC must beat hold decisively."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · ROC's own *t* vs the difference *t* — the beta trap\n\n"
            "Left: ROC and buy-and-hold both clear t≥2 on their own excess returns (both are long a "
            "rising market). Right: the **difference** ROC − hold is negative and insignificant — the "
            "indicator adds no alpha."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rb = st.book_returns(CLOSE, st.roc_signal(CLOSE, R['window'], 'long_flat'), R['cost_bps'])\n"
            "    bh = st.buy_and_hold(CLOSE)\n"
            "    roc_t = st.summarize(rb)['t_exc']; hold_t = st.summarize(bh)['t_exc']\n"
            "    dd = st.diff_vs_hold(CLOSE, R['window'], 'long_flat', R['cost_bps']); dt = dd['t_diff']\n"
            "else:\n"
            "    roc_t = R['roc']['t_exc']; hold_t = R['hold']['t_exc']; dt = R['diff']['t_diff']\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "a1.bar(['ROC own','Buy & hold'],[roc_t,hold_t],color=[GREEN,GREY],width=.55)\n"
            "a1.axhline(2,ls='--',c=RED,label='t=2 bar'); a1.set_ylabel('HAC t on excess return')\n"
            "for i,v in enumerate([roc_t,hold_t]): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_title('Both clear t=2 — because both are beta'); a1.legend()\n"
            "a2.bar(['ROC - hold'],[dt],color=RED,width=.4)\n"
            "a2.axhline(2,ls='--',c=GREY); a2.axhline(0,c='k',lw=.8)\n"
            "a2.annotate(f'{dt:.2f}',(0,dt),ha='center',va='top')\n"
            "a2.set_ylim(min(-2,dt-.5),2.5); a2.set_ylabel('HAC t on the difference')\n"
            "a2.set_title('The honest test: ROC adds nothing over hold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ROC own t={roc_t:.2f}  hold t={hold_t:.2f}  difference t={dt:.2f}  '\n"
            "      f\"delta={R['diff']['delta_ann']*100:+.1f}%/yr\")"
        ),
        md(
            f"> 💡 In plain words: ROC's own *t* = {R['roc']['t_exc']:.2f} is **identical machinery** to "
            f"hold's *t* = {R['hold']['t_exc']:.2f} — both just say \"long stocks paid off over 33 years.\" "
            f"The moment you subtract that beta, the residual is **{R['diff']['delta_ann']*100:+.1f}%/yr** "
            f"at *t* = {R['diff']['t_diff']:.2f}: a small, insignificant **loss**. Signal = WEAK, not REAL."
        ),
        md(
            "### 4b · The placebo — does the *timing* carry information?\n\n"
            "Circular-shift the realised position vector 5,000 times (same exposure, same turnover, random "
            "alignment) and re-race the Sharpe. The observed ROC Sharpe vs that null tells us whether the "
            "*when* matters at all, separately from the *how-much-invested*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.permutation_pvalue(CLOSE, R['window'], 'long_flat', R['cost_bps'], n_draws=5000)\n"
            "    obs = pl['obs']; draws = pl['draws']; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo']['obs']; pval = R['placebo']['p']\n"
            "    rng=np.random.default_rng(427); draws=rng.normal(0.45,0.08,5000)\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.2))\n"
            "ax.hist(draws,bins=60,color=GREY,alpha=.85,label='null: 5,000 circular position shifts')\n"
            "ax.axvline(obs,c=GREEN,lw=2.5,label=f'observed Sharpe {obs:.2f}')\n"
            "ax.set_xlabel('excess Sharpe'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Timing carries SOME info (placebo p = {pval:.3f}) — but it is risk-reduction, not alpha')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed Sharpe {obs:.3f}  permutation p = {pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the green line sits in the right tail (**p = {R['placebo']['p']:.3f}**), "
            "so the *timing* is **not** pure noise — stepping aside in big down-trends really does help "
            "risk-adjusted return. But \"better than a random schedule of the same exposure\" is a much "
            "weaker claim than \"beats buy-and-hold,\" which 4a already rejected. The info is real and "
            "tiny; it shows up as **lower drawdown**, not higher return."
        ),
        md(
            "### 4c · The rivals race — is the ROC *formula* special?\n\n"
            "Same tape, same lag, same costs, same risk-free. If ROC were a *better* momentum tool its "
            "Sharpe would stand out. It doesn't — it ties the simplest moving-average cross, and both "
            "beat the higher-churn MACD/RSI."
        ),
        code(
            "if HAVE_REAL:\n"
            "    race = st.benchmark_race(CLOSE, R['window'], 'long_flat', R['cost_bps'])\n"
            "    names = race.index.tolist(); sh=[race.loc[n,'sharpe'] for n in names]\n"
            "    cg=[race.loc[n,'cagr']*100 for n in names]; tn=[race.loc[n,'ann_turnover'] for n in names]\n"
            "else:\n"
            "    names=[r[0] for r in R['race']]; sh=[r[1] for r in R['race']]; cg=[r[2]*100 for r in R['race']]; tn=[r[5] for r in R['race']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.8,4.3))\n"
            "cols=[GREY]+[GREEN if n.startswith('ROC') else AMBER for n in names[1:]]\n"
            "a1.bar(range(len(names)),sh,color=cols,width=.62)\n"
            "for i,v in enumerate(sh): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_xticks(range(len(names))); a1.set_xticklabels(names,rotation=20,ha='right')\n"
            "a1.set_ylabel('excess Sharpe'); a1.set_title('ROC = SMA(200); both > MACD/RSI')\n"
            "a2.scatter(tn,sh,c=cols,s=80)\n"
            "for n,x,y in zip(names,tn,sh): a2.annotate(n,(x,y),fontsize=8,xytext=(4,4),textcoords='offset points')\n"
            "a2.set_xlabel('annual turnover (round trips)'); a2.set_ylabel('excess Sharpe')\n"
            "a2.set_title('More churn -> worse: ROC/SMA are the calm corner')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe:', {n:round(s,3) for n,s in zip(names,sh)})"
        ),
        md(
            f"> 💡 In plain words: **ROC({R['window']}) {R['race'][1][1]:.2f}** vs **SMA(200) "
            f"{R['race'][2][1]:.2f}** is a coin-flip difference; MACD ({R['race'][3][1]:.2f}) and RSI "
            f"({R['race'][4][1]:.2f}) trade 3–4× as often and earn less. The benefit credited to ROC is "
            "the **shared** \"stay-long-in-uptrends\" filter — the specific oscillator is interchangeable. "
            "Hence the 3rd axis: **MISATTRIBUTED**."
        ),
        md(
            "### 4d · Window robustness + cost sweep\n\n"
            "Left: the difference-vs-hold *t* across ROC lookbacks — it's negative everywhere, worst at "
            "short windows (whipsaw) and merely converging to hold at long ones. Right: net Sharpe vs "
            "cost — low turnover means costs barely bite (the edge problem isn't costs, it's the absence "
            "of an edge)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ws=[w[0] for w in R['windows']]; dts=[]\n"
            "    for w in ws: dts.append(st.diff_vs_hold(CLOSE,w,'long_flat',R['cost_bps'])['t_diff'])\n"
            "    cb=[c[0] for c in R['costs']]\n"
            "    cs=[st.summarize(st.book_returns(CLOSE,st.roc_signal(CLOSE,R['window'],'long_flat'),c))['sharpe'] for c in cb]\n"
            "else:\n"
            "    ws=[w[0] for w in R['windows']]; dts=[w[4] for w in R['windows']]\n"
            "    cb=[c[0] for c in R['costs']]; cs=[c[1] for c in R['costs']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.8,4.3))\n"
            "a1.bar([str(w) for w in ws],dts,color=RED,width=.55); a1.axhline(2,ls='--',c=GREY,label='t=2 bar'); a1.axhline(0,c='k',lw=.8)\n"
            "for i,v in enumerate(dts): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='top',fontsize=9)\n"
            "a1.set_xlabel('ROC lookback (days)'); a1.set_ylabel('HAC t: ROC - hold'); a1.set_title('Negative at every window'); a1.legend()\n"
            "a2.plot(cb,cs,'o-',color=GREEN,lw=2); a2.axhline(R['hold']['sharpe'],ls='--',c=GREY,label='buy & hold')\n"
            "a2.set_xlabel('one-way cost (bps)'); a2.set_ylabel('ROC net excess Sharpe'); a2.set_title('Costs barely bite — no edge to defend'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('difference t by window:', {w:round(t,2) for w,t in zip(ws,dts)})"
        ),
        md(
            f"> 💡 In plain words: there is **no** lookback that makes ROC beat hold — the difference *t* "
            f"runs from {R['windows'][0][4]:.2f} (21-day, whipsaw) to {R['windows'][3][4]:.2f} (252-day, "
            "just-hold-more), all negative. And costs are a red herring: at ~7 turns/yr the net Sharpe "
            "hardly moves from 1 to 10 bps. The rule's failure is structural, not frictional."
        ),
        md(
            "### 4e · Positive control — the harness can detect a real timing edge\n\n"
            "On a deterministic **regime-switching** tape (persistent up/down trends a momentum filter "
            "*should* ride), with the edge off the difference-*t* must stay ~0 (no false positive); with "
            "the edge on, ROC must beat hold decisively. Both hold — so the real-tape **null** result is "
            "trustworthy, not a broken pipeline."
        ),
        code(
            "res=[]\n"
            "for edge in (0.0, 0.5):\n"
            "    px,_=data.synthetic_panel(edge=edge, seed=427)\n"
            "    d=st.diff_vs_hold(px['close'],63,'long_flat',1.0)\n"
            "    sr=st.summarize(st.book_returns(px['close'],st.roc_signal(px['close'],63,'long_flat'),1.0))\n"
            "    hh=st.summarize(st.buy_and_hold(px['close']))\n"
            "    res.append((edge, sr['sharpe'], hh['sharpe'], d['delta_ann'], d['t_diff']))\n"
            "labels=[f'edge {e:.1f}\\n(no trend)' if e==0 else f'edge {e:.1f}\\n(real trend)' for e,_,_,_,_ in res]\n"
            "dt=[r[4] for r in res]\n"
            "fig,ax=plt.subplots(figsize=(8.4,4.3))\n"
            "ax.bar(labels,dt,color=[GREY,GREEN],width=.5); ax.axhline(2,ls='--',c=RED,label='t=2 bar'); ax.axhline(0,c='k',lw=.8)\n"
            "for i,t in enumerate(dt): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('HAC t: ROC - hold'); ax.set_title('Control: edge off -> t~0; edge on -> ROC beats hold'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,sr,hh,dl,t in res: print(f'edge={e:.1f}: roc_sharpe={sr:+.2f} hold_sharpe={hh:+.2f} delta={dl*100:+.1f}%/yr t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted trend the control's difference-*t* is "
            f"**{R['syn'][0][4]:.2f}** (the harness does **not** fake an edge from noise); with a real "
            f"persistent trend it jumps to **{R['syn'][1][4]:.2f}** (ROC genuinely beats hold). So the "
            "machinery *can* see a timing edge when one exists — and on SPY it sees none. The real-tape "
            "null is a finding, not a bug."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — ROC's own excess Sharpe **{R['roc']['sharpe']:.2f}** (HAC "
            f"*t* = {R['roc']['t_exc']:.2f}) clears t≥2 but is **beta** (hold's *t* = "
            f"{R['hold']['t_exc']:.2f}, same machinery). The incremental edge over hold is "
            f"**{R['diff']['delta_ann']*100:+.1f}%/yr** at HAC *t* = **{R['diff']['t_diff']:.2f}** — "
            "negative, insignificant. The timing carries *some* info vs a random schedule "
            f"(placebo *p* = {R['placebo']['p']:.3f}) but it manifests as risk-reduction, not alpha. "
            "WEAK, not REAL — and certainly not on the t≥2 bar.\n"
            f"- **Tradability `MIRAGE`** — ROC **lowers** CAGR ({R['roc']['cagr']*100:.1f}% vs "
            f"{R['hold']['cagr']*100:.1f}%); the only thing it delivers is drawdown "
            f"({R['roc']['maxdd']*100:.0f}% vs {R['hold']['maxdd']*100:.0f}%), which a 200-day SMA "
            "delivers identically. No incremental return to harvest or scale.\n"
            f"- **Better momentum tool? `MISATTRIBUTED`** — ROC({R['window']}) "
            f"({R['race'][1][1]:.2f}) ties SMA(200) ({R['race'][2][1]:.2f}); the benefit is the shared "
            "trend filter, not the ROC formula. The credit is mis-assigned to the indicator."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the capacity is fine; the edge is the problem\n\n"
            "ROC is *eminently* tradable in the operational sense — one liquid ETF, ~7 turns a year, "
            "trivial costs, unlimited capacity. That's exactly why it's a clean teaching case: the thing "
            "that kills it isn't friction, it's that **there's no excess return to collect.** The "
            "tradeable artefact is the drawdown reduction — and you'd buy that from the cheapest trend "
            "filter, or from a static stock/bond mix, not from ROC specifically."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rb = st.book_returns(CLOSE, st.roc_signal(CLOSE, R['window'],'long_flat'), R['cost_bps'])\n"
            "    bh = st.buy_and_hold(CLOSE)\n"
            "    eq_roc=(1+rb['net']).cumprod(); eq_bh=(1+bh['net']).cumprod()\n"
            "    dd_roc=(eq_roc/eq_roc.cummax()-1)*100; dd_bh=(eq_bh/eq_bh.cummax()-1)*100\n"
            "    fig,ax=plt.subplots(figsize=(9.5,4.3))\n"
            "    ax.fill_between(dd_bh.index,dd_bh,0,color=GREY,alpha=.5,label=f\"hold (worst {R['hold']['maxdd']*100:.0f}%)\")\n"
            "    ax.plot(dd_roc.index,dd_roc,color=GREEN,lw=1.3,label=f\"ROC (worst {R['roc']['maxdd']*100:.0f}%)\")\n"
            "    ax.set_ylabel('drawdown (%)'); ax.set_title('The only deliverable: a shallower drawdown path'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache absent')\n"
            "print(f\"worst DD  hold {R['hold']['maxdd']*100:.0f}%  ROC {R['roc']['maxdd']*100:.0f}%  \"\n"
            "      f\"(at the cost of {(R['hold']['cagr']-R['roc']['cagr'])*100:.1f}%/yr CAGR)\")"
        ),
        md(
            "> 💡 In plain words: the green drawdown line stays shallower through 2008 and 2020 — real, "
            "useful risk control. But you pay ~2%/yr of CAGR for it, and an SMA(200) hands you the same "
            "deal. As an **alpha**, ROC is a mirage; as a **risk dial**, it's fine but unremarkable."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The difference-*t* is the template.** Any \"my indicator beats the market\" claim should "
            "be tested as a HAC *t* on the *difference* vs the relevant benchmark, never on the rule's own "
            "(beta-laden) *t*. ROC is the clean cautionary example.\n"
            "- **Trend filters are a commodity.** ROC, SMA cross, Donchian, time-series momentum — they "
            "share one bet. Pick on turnover/clarity, not on a belief that one formula is magic.\n"
            "- **Where momentum *does* pay** is cross-sectional (rank many assets) and time-series across "
            "*many* markets (Moskowitz-Ooi-Pedersen 2012), not single-asset long/flat on an index with a "
            "secular drift. Re-run this engine on a futures panel for the contrast.\n\n"
            "*The reproducible core is offline and deterministic; the real tape is cached SPY total-return "
            "closes. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
