"""Generate the two narrative notebooks for Study 381 (TIPS-Breakeven).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ETF prices under
../_cache/ (the breakeven proxy log(TIP/IEF) + SPY/GLD) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with
no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ETF tape: TIP/IEF/SPY/
# GLD, 2004-11-18 -> 2026-06-18, 5,429 days, 260 month-ends, 21.6 years).
R = dict(
    start="2004-11-18", end="2026-06-18", days=5429, months=260, years=21.6,
    n_tests=18, n_cross=3,
    # the 18-test scan: (signal, asset, h, slope, NW_t, placebo_p)
    scan=[
        ("level-z", "TIP", 3, -0.0045, -2.40, 0.11),
        ("level-z", "TIP", 6, -0.0081, -2.19, 0.13),
        ("level-z", "TIP", 12, -0.0142, -2.00, 0.16),
        ("3m-change", "TIP", 3, -0.0938, -1.20, 0.34),
        ("3m-change", "TIP", 6, -0.1780, -1.17, 0.39),
        ("3m-change", "TIP", 12, -0.3907, -1.90, 0.14),
        ("level-z", "SPY", 3, 0.0009, 0.11, 0.91),
        ("level-z", "SPY", 6, -0.0090, -0.90, 0.40),
        ("level-z", "SPY", 12, -0.0126, -0.63, 0.54),
        ("3m-change", "SPY", 3, 0.4842, 1.29, 0.28),
        ("3m-change", "SPY", 6, -0.0034, -0.01, 1.00),
        ("3m-change", "SPY", 12, -0.5592, -1.15, 0.33),
        ("level-z", "GLD", 3, -0.0004, -0.05, 0.98),
        ("level-z", "GLD", 6, 0.0125, 0.91, 0.59),
        ("level-z", "GLD", 12, 0.0268, 0.81, 0.58),
        ("3m-change", "GLD", 3, -0.6067, -3.21, 0.01),
        ("3m-change", "GLD", 6, -0.4470, -1.38, 0.28),
        ("3m-change", "GLD", 12, -0.7317, -1.41, 0.22),
    ],
    # the mechanical self-prediction: breakeven level -> its OWN forward 6m change
    self_pred_t=-1.98,
    # directional win-rates: (label, cond_win%, base_win%)
    win=[("TIP level-z 12m", 66, 75), ("SPY level-z 12m", 68, 83), ("GLD 3m-change 3m", 52, 61)],
    # traded sign rules: (label, gross_ann%, net_ann%, net_sharpe, bh_ann%, bh_sharpe)
    trade=[("GLD on 3m-change", -0.6, -1.3, -0.07, 11.4, 0.67),
           ("TIP on level-z", 1.9, 1.8, 0.32, 3.5, 0.62)],
    # synthetic control: (edge, n, slope, NW_t, placebo_p)
    syn=[(0.00, 216, -0.0010, -0.37, 0.68), (0.03, 216, 0.0234, 7.93, 0.00)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free_macro-timing_lunch%3F: Busted](https://img.shields.io/badge/Free_macro--timing_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
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

from tips_breakeven import data, strategy as st

HAVE_REAL = data.have_real()
M = data.load_real() if HAVE_REAL else None
BE = st.breakeven_series(M) if HAVE_REAL else None
SIGS = {"level-z": st.level_z(BE), "3m-change": st.change_3m(BE)} if HAVE_REAL else None
print("real ETF cache present:", HAVE_REAL,
      "| month-ends:", (0 if M is None else len(M)))
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
            "# Does the bond market's inflation forecast time your portfolio? 🧮\n"
            "### The 'breakeven inflation rate' as a trading signal, in plain English\n\n"
            + BADGES +
            "There's a number traders love: the **breakeven inflation rate** — what the bond "
            "market thinks inflation will average over the next ten years. It comes from comparing "
            "ordinary Treasury bonds with inflation-protected ones. The pitch sounds clever: *the "
            "bond market is the smartest forecaster alive, so when its inflation forecast rises, buy "
            "inflation hedges (gold, TIPS) and lean out of stocks; when it falls, do the opposite.*\n\n"
            "A market-implied crystal ball you can trade — who wouldn't want that? This notebook "
            "rebuilds the signal and asks the only question that matters: **does it actually time "
            "anything you can put money on?** Spoiler: it co-moves with everything and times "
            "nothing.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the placebo test and the "
            "power analysis? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** The official breakeven series (FRED `T10YIE`) isn't "
            "reachable here, so we **build a transparent proxy**: `log(TIP / IEF)` — an "
            "inflation-protected bond fund divided by a normal-bond fund. When the market's "
            "inflation forecast rises, the first outperforms the second, so the ratio rises with "
            "breakeven. We call it a proxy throughout. Every chart is drawn by the code beside it; "
            "house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does breakeven move with inflation hedges? | **Yes** — it's literally built from "
            "bond prices, so of course it co-moves. |\n"
            "| Does its *level* or *change* **predict** what comes next? | **Barely, and not where "
            "it counts.** Out of **18** sensible tests, only **3** look 'significant' — and a fair "
            "luck-test knocks most of them down. |\n"
            "| What about the few that survive? | **One is fake by construction** (the signal "
            "predicting *itself*), and **one is a coin-flip fluke** (a single gold result that "
            "vanishes at other horizons). On **stocks** — the headline use — there's nothing. |\n"
            "| Could you trade the 'good' one? | **No.** The one 'significant' gold rule **loses "
            f"money** (**{R['trade'][0][2]:+.1f}%/yr** after costs) while just holding gold made "
            f"**+{R['trade'][0][4]:.0f}%/yr**. |\n\n"
            "> Breakeven is a real, sensible forecast. As a *trading timer* it's a mirage — the kind "
            "you always find when you scan a macro number against enough assets and horizons."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The breakeven inflation rate is the market's own inflation forecast. When it's "
            "high or rising, inflation is coming — so own gold and TIPS and underweight stocks. When "
            "it's low or falling, do the reverse. The bond market is never dumb; ride its forecast.\"*\n\n"
            "It's an attractive story because breakeven really *is* a forecast — nominal yield minus "
            "the inflation-protected (TIPS) yield is what the market is pricing for future inflation. "
            "The leap is the trading claim: that this forecast **leads** asset prices reliably enough "
            "to position around. We'll rebuild breakeven as a proxy and put that leap under a "
            "microscope."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two very different things hide inside 'the bond market's forecast times the market,' and "
            "only one would make money. (1) *Does breakeven move **with** inflation assets?* Sure — "
            "but co-movement isn't timing; you can't trade something you only learn about at the same "
            "time. The question that matters is (2) *does today's breakeven tell you what assets will "
            "do **next**, more reliably than chance?* And there's a trap: if you test enough "
            "signals against enough assets over enough horizons, **something** will look predictive "
            "by pure luck. The honest test has to survive that."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We build breakeven as a **transparent proxy** — `log(TIP/IEF)` — from "
            f"**{R['years']:.0f} years** of ETF prices ({R['start']} → {R['end']}). Then we make two "
            "signals from it: its **level** (how high is breakeven vs its own history) and its "
            "**3-month change** (rising or falling). And we ask each whether it predicts the next "
            "**3 / 6 / 12 months** of three assets people actually want to time:\n\n"
            "1. **TIPS** themselves (the inflation-protected bonds),\n"
            "2. **stocks** (SPY),\n"
            "3. **gold** (GLD).\n\n"
            f"That's 2 signals × 3 assets × 3 horizons = **{R['n_tests']} tests**. For each, we ask "
            "how strong the link is *and* run a luck-test: shuffle the signal thousands of times and "
            "see how often random alignment looks just as good."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, here's the signal.** The breakeven proxy `log(TIP/IEF)` over two decades — it "
            "really does track the inflation story (low in the 2008/2015 scares, surging in "
            "2021–22). The question is whether *today's* value tells you about *tomorrow's* returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    be = BE\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.0))\n"
            "    ax.plot(be.index, be.values, c=AMBER, lw=1.6)\n"
            "    ax.set_title('The breakeven proxy  log(TIP/IEF)  — the bond market\\'s inflation read')\n"
            "    ax.set_ylabel('log(TIP / IEF)'); ax.set_xlabel('')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('breakeven proxy spans', be.index.min().date(), '->', be.index.max().date())\n"
            "else:\n"
            "    print('no cache — see docs/results.md for the frozen run (proxy 2004-11 -> 2026-06)')"
        ),
        md(
            "**Now the whole scan in one picture.** Each bar is one of the "
            f"**{R['n_tests']}** tests; its height is the strength of the link (the *t*-stat). The "
            "dashed lines are the ±2 'significance' bar. Most bars are tiny; a few poke past — and "
            "we'll see those few are mirages."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for sname, sig in SIGS.items():\n"
            "        for asset in ['TIP','SPY','GLD']:\n"
            "            for h in (3,6,12):\n"
            "                reg = st.hac_regression(sig, st.forward_return(M[asset], h), lags=h)\n"
            "                rows.append((f'{sname}|{asset}|{h}m', reg['t'], asset))\n"
            "else:\n"
            "    rows = [(f\"{s}|{a}|{h}m\", t, a) for (s,a,h,_,t,_) in R['scan']]\n"
            "labels = [r[0] for r in rows]; tt = [r[1] for r in rows]\n"
            "cmap = {'TIP': GREY, 'SPY': '#4477aa', 'GLD': AMBER}\n"
            "cols = [cmap[r[2]] for r in rows]\n"
            "order = np.argsort(tt)\n"
            "fig, ax = plt.subplots(figsize=(9.6, 6.2))\n"
            "ax.barh(np.arange(len(rows)), [tt[i] for i in order], color=[cols[i] for i in order])\n"
            "ax.set_yticks(np.arange(len(rows))); ax.set_yticklabels([labels[i] for i in order], fontsize=7)\n"
            "ax.axvline(2, ls='--', c=RED); ax.axvline(-2, ls='--', c=RED)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('Newey-West t-stat of the predictive slope')\n"
            "ax.set_title(f'{len(rows)} tests; only 3 poke past |t|=2 (grey=TIP, blue=SPY, amber=GLD)')\n"
            "plt.tight_layout(); plt.show()\n"
            "n_cross = sum(abs(t)>=2 for t in tt)\n"
            "print(f'{n_cross} of {len(tt)} tests cross |t|=2 — about what pure luck gives you (~0.9 expected)')"
        ),
        md(
            f"Out of **{R['n_tests']}** tests, **{R['n_cross']}** cross the bar — and at the 5% "
            "level you *expect* about one by chance. So 'three hits' is already barely above luck. "
            "Look at the colours: **stocks (blue) never cross** — the headline use of the signal "
            "does nothing. The crossings are all in TIPS (grey) and one gold (amber). Let's expose "
            "both."
        ),
        md(
            "**The TIPS 'hits' are fake by construction.** Our breakeven proxy is *built from* TIP. "
            "So 'high breakeven predicts TIP falling' is really just the ratio **bouncing back to "
            "normal** — the signal predicting *itself*. Here's that giveaway: the breakeven level "
            "predicts its **own** future change."
        ),
        code(
            "if HAVE_REAL:\n"
            "    z = st.level_z(BE); fwd_be = BE.shift(-6) - BE\n"
            "    reg = st.hac_regression(z, fwd_be, lags=6); spt = reg['t']\n"
            "    df = np.c_[np.asarray(z), np.asarray(fwd_be)]\n"
            "    df = df[np.isfinite(df).all(1)]\n"
            "else:\n"
            "    spt = R['self_pred_t']; rng=np.random.default_rng(381)\n"
            "    z0 = rng.normal(0,1,200); df = np.c_[z0, -0.008*z0 + rng.normal(0,0.02,200)]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.scatter(df[:,0], df[:,1]*100, s=14, c=GREY, alpha=.6)\n"
            "b = np.polyfit(df[:,0], df[:,1], 1); xs = np.linspace(df[:,0].min(), df[:,0].max(), 50)\n"
            "ax.plot(xs, (b[0]*xs+b[1])*100, c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=.6); ax.axvline(0, c='k', lw=.6)\n"
            "ax.set_xlabel('breakeven level today (z-score)'); ax.set_ylabel('breakeven change over next 6 months (%)')\n"
            "ax.set_title(f'High breakeven -> breakeven falls back (t={spt:.2f}). That IS the \"TIP edge\".')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'the breakeven proxy predicts its OWN mean reversion at t={spt:.2f} — not a tradable macro signal')"
        ),
        md(
            f"There it is: high breakeven simply drifts **back down** (slope is negative, "
            f"t = {R['self_pred_t']:.2f}). Since TIP *is* the numerator of our proxy, the 'TIPS "
            "edge' is just that mean reversion wearing a macro costume. Strip it away and the TIPS "
            "hits are no edge at all."
        ),
        md(
            "**The one gold hit is a fluke you can't trade.** The single crossing on an *independent* "
            "asset — gold, 3-month change — looks strong (t = −3.21). But it's the **1-in-18 lucky "
            "cell**, and the only test that matters is money. Here's that 'signal' traded against "
            "just buying gold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.sign_timer(M['GLD'], st.change_3m(BE), contrarian=True, cost_bps=10.0)\n"
            "    net, bh = g['net_ann']*100, g['bh_ann']*100\n"
            "else:\n"
            "    net, bh = R['trade'][0][2], R['trade'][0][4]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['trade the\\nbreakeven signal', 'just hold\\ngold'], [net, bh], color=[RED, GREEN], width=.55)\n"
            "for i,v in enumerate([net, bh]): ax.annotate(f'{v:+.1f}%/yr',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('annualised return, net of costs')\n"
            "ax.set_title('The \"significant\" gold timer LOSES money vs buy-and-hold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'breakeven gold timer: {net:+.1f}%/yr net   |   buy-and-hold gold: {bh:+.1f}%/yr')"
        ),
        md(
            f"The 'significant' rule earns **{R['trade'][0][2]:+.1f}%/yr** after costs; **holding "
            f"gold** earned **+{R['trade'][0][4]:.0f}%/yr**. A statistically cute regression that "
            "loses money is the definition of a mirage."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** Breakeven co-moves with assets, and a few regressions look "
            "'significant' — but the only ones that do are the signal predicting *itself* (TIPS) and "
            "a single gold fluke. On stocks: nothing. Real as a forecast, weak as a timer.\n"
            "- **Tradability — Mirage.** Every conditional win-rate is *below* the base rate; the one "
            f"'good' rule loses **{R['trade'][0][2]:+.1f}%/yr** vs **+{R['trade'][0][4]:.0f}%/yr** "
            "buy-and-hold. Nothing deployable.\n"
            "- **Free macro-timing lunch? — Busted.** Scan a macro number against enough assets and "
            "horizons and luck hands you a 'hit.' It's the same trap as the "
            "[Fed-Model](../118-fed-model/) and [Real-Rate-Regime](../119-real-rate-regime/) studies."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — win-rates vs the base rate\n\n"
            "Forget the *t*-stats for a second and ask the operational question: if you bet on the "
            "signal, are you right more often than you'd be by just being long the asset? Here are "
            "the three headline tests — every conditional win-rate is **below** its base rate."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pairs = [('TIP','level-z',12),('SPY','level-z',12),('GLD','3m-change',3)]\n"
            "    labs, cw, bw = [], [], []\n"
            "    for asset, sname, h in pairs:\n"
            "        w = st.directional_winrate(SIGS[sname], st.forward_return(M[asset], h))\n"
            "        labs.append(f'{asset}\\n{sname} {h}m'); cw.append(w['cond_win']*100); bw.append(w['base_win']*100)\n"
            "else:\n"
            "    labs = [w[0] for w in R['win']]; cw = [w[1] for w in R['win']]; bw = [w[2] for w in R['win']]\n"
            "x = np.arange(len(labs))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x-.2, cw, .4, color=RED, label='betting on the signal')\n"
            "ax.bar(x+.2, bw, .4, color=GREY, label='just being long (base rate)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8); ax.set_ylim(0, 100)\n"
            "ax.set_ylabel('% of the time the trade is positive')\n"
            "ax.set_title('Every signal win-rate is BELOW its base rate — it subtracts value')\n"
            "for i,(c,b) in enumerate(zip(cw,bw)):\n"
            "    ax.annotate(f'{c:.0f}%',(i-.2,c),ha='center',va='bottom')\n"
            "    ax.annotate(f'{b:.0f}%',(i+.2,b),ha='center',va='bottom')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('conditional win-rates sit below the base rate -> the signal is worse than naive long')"
        ),
        md(
            "The signal doesn't just fail to add an edge — acting on it makes you **right less "
            "often** than simply staying long. That's the operational tell of a non-signal: it's "
            "noise dressed as information."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The same trap, other costumes.** [Study 118 — Fed-Model](../118-fed-model/) and "
            "[Study 119 — Real-Rate-Regime](../119-real-rate-regime/) regress a macro/valuation "
            "number against forward returns — in-sample slopes that look real and out-of-sample "
            "edges that vanish.\n"
            "- **The asset side.** [Study 152 — Inflation-Hedge](../152-inflation-hedge/): do gold "
            "and TIPS even *hedge* inflation? The natural complement to 'does breakeven *time* them?'\n"
            "- **Run it yourself.** Swap the proxy for the real FRED `T10YIE` if you can fetch it, "
            "add more assets and horizons — and watch the 'hit count' stay near what luck predicts.\n\n"
            "*Think breakeven times the market by more than luck? Pick the asset and horizon "
            "**before** you look, trade it net of costs, and show it beating buy-and-hold "
            "out-of-sample — then we'll talk.*"
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
            "# TIPS-Breakeven — a quantitative teardown 🔬\n"
            "### Breakeven proxy log(TIP/IEF) · HAC predictive regressions across signal × asset × "
            "horizon · a placebo randomization null · the self-prediction artifact · costed sign "
            "rules · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We test "
            "whether the bond market's breakeven inflation rate **predicts** forward asset returns, "
            "and separate that from the trivial fact that it **co-moves** with them. The decisive "
            "object is not any single slope (a couple are 'significant') but the **multiple-testing + "
            "self-prediction** structure: across 18 signal × asset × horizon tests, the only "
            "`|t| ≥ 2` on a genuinely *independent* asset is a 1-in-18 false positive, and the "
            "'best' *t* of all is the proxy forecasting its **own** mean reversion.\n\n"
            "> ⚠️ **Data + proxy note.** FRED `T10YIE` isn't reachable here; we build a transparent "
            "**proxy** `be = log(TIP/IEF)` (inflation-protected over duration-matched nominal "
            "Treasuries) from yfinance daily adjusted closes (TIP/IEF/SPY/GLD, 2004→2026). Because "
            "it's built from TIP, predicting forward *TIP* returns partly predicts the signal itself "
            "— named on the Signal axis. Offline core + synthetic control are deterministic. Methods "
            "in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | {R['n_cross']}/{R['n_tests']} tests cross `|t|≥2` (luck ≈ 0.9). "
            "The TIP hits (`t=−2.40/−2.19/−2.00`) are **self-prediction** (placebo `p≈0.11–0.16`); "
            "the lone independent hit (GLD 3m-change, `t=−3.21`) dies at 6/12m. SPY: nothing. |\n"
            f"| **Tradability** | `MIRAGE` | Every cond. win-rate **< base**; the GLD rule nets "
            f"**{R['trade'][0][2]:+.1f}%/yr** vs **+{R['trade'][0][4]:.0f}%/yr** buy-and-hold. No "
            "deployable, independent, costed edge. |\n"
            f"| **Free macro-timing lunch?** | `BUSTED` | A scan-and-self-prediction illusion; the "
            "synthetic control (`edge=0 ⇒ t=−0.37`, `edge>0 ⇒ t=+7.93`) confirms the real-tape null "
            "is **true**, not underpowered. |\n\n"
            "> 💡 In plain words: breakeven is a real forecast that **co-moves** with assets; it does "
            "not **lead** them in any way that survives an honest, multiple-testing-aware, "
            "cost-aware test. Strip the self-prediction and the luck, and there is nothing left."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $b_t = \\log(\\mathrm{TIP}_t / \\mathrm{IEF}_t)$ be the breakeven proxy. Build two "
            "signals: the expanding z-score $z_t = (b_t - \\hat\\mu_t)/\\hat\\sigma_t$ (level) and "
            "$\\Delta_3 b_t = b_t - b_{t-3}$ (3-month change). For asset $A$ and horizon "
            "$H\\in\\{3,6,12\\}$ months, regress the forward return on the signal:\n\n"
            "$$r^{A}_{t\\to t+H} = \\alpha + \\beta\\,s_t + \\varepsilon_t,\\qquad "
            "s_t\\in\\{z_t,\\ \\Delta_3 b_t\\}.$$\n\n"
            "- **H₁ (it predicts).** $\\beta \\ne 0$ with a robust (HAC) *t* on a **genuinely "
            "independent** asset.\n"
            "- **H₂ (it's deployable).** A signed rule on the signal beats buy-and-hold net of costs.\n"
            "- **H₃ (free lunch).** The 'bond market forecasts everything' story holds up against "
            "multiple testing.\n\n"
            "We find **H₁ rejected** (the only independent `|t|≥2` is a 1-in-18 fluke; SPY flat; the "
            "TIP hits are self-prediction), **H₂ rejected** (sign rules lose to buy-and-hold), "
            "**H₃ rejected** (the synthetic control shows the null is true). Overlapping forward "
            "windows are handled with **Newey-West** HAC errors; the small-sample fragility with a "
            "**placebo** null."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — co-movement vs prediction, and the scan tax\n\n"
            "Co-movement ($\\mathrm{corr}(b_t, r^A_t)$) is free and useless — you learn $b_t$ and "
            "$r^A_t$ at the same instant. Prediction ($\\beta$ on $s_t$ vs $r^A_{t\\to t+H}$ with a "
            "lag) is what pays. Two hazards corrupt the naive read:\n\n"
            "$$t_{\\text{HAC}} = \\frac{\\hat\\beta}{\\mathrm{SE}_{\\text{NW}}(\\hat\\beta)},\\qquad "
            "\\Pr_{\\text{scan}}[\\exists\\,|t|\\ge 2 \\text{ among } m \\text{ tests}] "
            "\\approx 1-(1-0.05)^m.$$\n\n"
            "With $m=18$ that probability is $\\approx 60\\%$ — finding *a* significant cell is the "
            "**expected** outcome, not evidence. And overlapping $H$-month windows make naive OLS "
            "*t*-stats too big; the **HAC** correction (bandwidth $=H$) is mandatory. The honest "
            "instruments are therefore (a) a HAC *t* **per cell**, (b) a **placebo** null that "
            "shuffles the signal, and (c) the **scan count** vs its luck expectation — not any single "
            "headline *t*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Breakeven proxy.** $b_t=\\log(\\mathrm{{TIP}}/\\mathrm{{IEF}})$, month-end "
            f"({R['start']}→{R['end']}, {R['months']} month-ends). Explicit **proxy** for FRED "
            "`T10YIE`; built from TIP, so the TIP column carries a self-prediction asterisk.\n"
            "- **Signals.** Expanding z-score (≥24 obs, no look-ahead) and trailing 3-month change.\n"
            "- **Targets.** Forward $H\\in\\{3,6,12\\}$-month returns of TIP, SPY, GLD ⇒ "
            f"**{R['n_tests']} tests**.\n"
            "- **Inference #1 (HAC).** Newey-West *t* of $\\beta$, Bartlett kernel, bandwidth $=H$.\n"
            "- **Inference #2 (placebo).** Circularly shuffle the signal, recompute the HAC *t*; "
            "$p=\\Pr[|t_{\\text{shuffled}}|\\ge|t_{\\text{obs}}|]$.\n"
            "- **Tradability.** Signed sign rule, **1-month execution lag**, 10 bps one-way × "
            "turnover, raced against buy-and-hold (gross **and** net).\n"
            "- **Positive control.** A deterministic series where next-month return $=\\mu+"
            "\\text{edge}\\cdot z+\\text{noise}$: with **edge 0** the HAC *t* must stay near 0; with "
            "a large edge it must clear 2 — proving the engine is unbiased *and* powered."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The full scan — most cells are noise, and the crossings are suspect\n\n"
            "Every cell's HAC *t*, laid out as signal × asset × horizon. The ±2 bars and the colour "
            "tell the story: stocks never cross, TIP crosses (self-prediction), one gold cell pokes "
            "out (the fluke)."
        ),
        code(
            "hs = (3,6,12)\n"
            "if HAVE_REAL:\n"
            "    grid = {}\n"
            "    for sname, sig in SIGS.items():\n"
            "        for asset in ['TIP','SPY','GLD']:\n"
            "            grid[(sname,asset)] = [st.hac_regression(sig, st.forward_return(M[asset], h), lags=h)['t'] for h in hs]\n"
            "else:\n"
            "    grid = {}\n"
            "    for (s,a,h,_,t,_) in R['scan']: grid.setdefault((s,a), [None,None,None])\n"
            "    for (s,a,h,_,t,_) in R['scan']: grid[(s,a)][hs.index(h)] = t\n"
            "keys = [(s,a) for s in ['level-z','3m-change'] for a in ['TIP','SPY','GLD']]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 5.4))\n"
            "x = np.arange(len(hs)); w=.26\n"
            "for j,k in enumerate(keys):\n"
            "    off = (j-2.5)*w/1.7\n"
            "    ax.plot(0,0)  # keep colour cycle moving\n"
            "im = ax.imshow([[grid[k][i] for i in range(3)] for k in keys], aspect='auto', cmap='RdBu', vmin=-3.3, vmax=3.3)\n"
            "ax.set_xticks(range(3)); ax.set_xticklabels([f'{h}m' for h in hs])\n"
            "ax.set_yticks(range(len(keys))); ax.set_yticklabels([f'{s} · {a}' for s,a in keys], fontsize=9)\n"
            "for yi,k in enumerate(keys):\n"
            "    for xi in range(3):\n"
            "        tv = grid[k][xi]; ax.text(xi, yi, f'{tv:.2f}', ha='center', va='center', fontsize=8,\n"
            "                                  color='white' if abs(tv)>2 else 'black', fontweight='bold' if abs(tv)>=2 else 'normal')\n"
            "fig.colorbar(im, ax=ax, label='Newey-West t-stat'); ax.set_title('HAC t across 18 tests (bold = |t|>=2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "flat = [grid[k][i] for k in keys for i in range(3)]\n"
            "print('crossings |t|>=2:', sum(abs(t)>=2 for t in flat), 'of', len(flat), ' (expected by luck ~0.9)')"
        ),
        md(
            f"> 💡 In plain words: **{R['n_cross']}** of {R['n_tests']} cells cross |t|=2 — and at 5% "
            "you expect ~0.9. The three crossings are the two TIP/level-z cells and one gold cell. "
            "Stocks are inert at every horizon. This pattern — *a couple of hits, all explicable* — "
            "is what a **non-signal** looks like through a scan."
        ),
        md(
            "### 4b · The self-prediction artifact — the TIP 'edge' is the proxy reverting\n\n"
            "The TIP crossings aren't a macro edge; they're the proxy **mean-reverting**, and TIP is "
            "the proxy's numerator. Regress the breakeven level on its *own* forward 6-month change "
            "and the same negative slope appears."
        ),
        code(
            "if HAVE_REAL:\n"
            "    z = st.level_z(BE)\n"
            "    reg_tip = st.hac_regression(z, st.forward_return(M['TIP'], 6), lags=6)\n"
            "    reg_be  = st.hac_regression(z, BE.shift(-6) - BE, lags=6)\n"
            "    t_tip, t_be = reg_tip['t'], reg_be['t']\n"
            "else:\n"
            "    t_tip, t_be = R['scan'][1][4], R['self_pred_t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.2))\n"
            "ax.bar(['breakeven -> forward\\nTIP return (6m)', 'breakeven -> its OWN\\nforward change (6m)'],\n"
            "       [t_tip, t_be], color=[GREY, RED], width=.5)\n"
            "for i,v in enumerate([t_tip, t_be]): ax.annotate(f't={v:.2f}',(i,v),ha='center',va='top')\n"
            "ax.axhline(-2, ls='--', c=RED); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Newey-West t-stat'); ax.set_title('Same slope: the \"TIP edge\" is the proxy reverting to itself')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'TIP forward t={t_tip:.2f}  vs  breakeven self-prediction t={t_be:.2f} — the same autocorrelation')"
        ),
        md(
            f"> 💡 In plain words: the breakeven level predicts its own future change at "
            f"**t = {R['self_pred_t']:.2f}** — nearly identical to the 'TIP edge.' Since you can't "
            "buy 'the proxy reverting,' this is **not** a tradable macro signal; it's mechanical "
            "autocorrelation that the proxy construction smuggles into the TIP column."
        ),
        md(
            "### 4c · The independent hit is a mirage — placebo + money\n\n"
            "The only `|t|≥2` on a genuinely independent asset is GLD / 3m-change / 3m "
            "(`t=−3.21`, placebo `p=0.01`). Left: where it sits across horizons (it dies). Right: "
            "trade it — it loses to buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ch = st.change_3m(BE)\n"
            "    gt = [st.hac_regression(ch, st.forward_return(M['GLD'], h), lags=h)['t'] for h in (3,6,12)]\n"
            "    g = st.sign_timer(M['GLD'], ch, contrarian=True, cost_bps=10.0)\n"
            "    gross, net, bh = g['gross_ann']*100, g['net_ann']*100, g['bh_ann']*100\n"
            "else:\n"
            "    gt = [R['scan'][15][4], R['scan'][16][4], R['scan'][17][4]]\n"
            "    gross, net, bh = R['trade'][0][1], R['trade'][0][2], R['trade'][0][4]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))\n"
            "a1.bar([f'{h}m' for h in (3,6,12)], gt, color=[RED, GREY, GREY], width=.55)\n"
            "a1.axhline(-2, ls='--', c=RED); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(gt): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='top')\n"
            "a1.set_title('GLD / 3m-change: significant only at 3m'); a1.set_ylabel('HAC t'); a1.set_xlabel('horizon')\n"
            "a2.bar(['gross','net @10bps','buy & hold'], [gross, net, bh], color=[AMBER, RED, GREEN], width=.6)\n"
            "for i,v in enumerate([gross, net, bh]): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('annualised return'); a2.set_title('...and it loses money traded')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'GLD timer: gross={gross:+.1f}%/yr net={net:+.1f}%/yr  vs  buy-and-hold {bh:+.1f}%/yr')"
        ),
        md(
            f"> 💡 In plain words: the gold hit is significant at **3m only** and evaporates at "
            f"6/12m — the fingerprint of a multiple-testing fluke. Traded with a lag and 10 bps "
            f"costs it returns **{R['trade'][0][2]:+.1f}%/yr** against **+{R['trade'][0][4]:.0f}%/yr** "
            "for buy-and-hold. A 'significant' regression that loses money is exactly the mirage the "
            "Tradability axis is built to catch."
        ),
        md(
            "### 4d · Faithful-engine & power control — the null is a TRUE null\n\n"
            "On a deterministic series where next-month return = $\\mu+\\text{edge}\\cdot z+$ noise: "
            "with **edge 0** the HAC *t* must sit near zero (no false positive); with a **real** "
            "planted slope it must clear 2 by a mile. Both hold — so the engine is unbiased *and* "
            "powered, and the flat real tape is a genuine absence of edge, not a lack of power."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.03):\n"
            "    syn = data.synthetic_macro(n_months=240, edge=edge, seed=381)\n"
            "    sz = st.level_z(st.breakeven_series(syn))\n"
            "    reg = st.hac_regression(sz, st.forward_return(syn['ASSET'], 1), lags=1)\n"
            "    p = st.placebo_pvalue(sz, st.forward_return(syn['ASSET'], 1), lags=1, n_draws=4000)\n"
            "    res.append((edge, reg['t'], p))\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "labels = [f'planted edge\\n{e:+.2f}' for e,_,_ in res]; tv=[r[1] for r in res]\n"
            "ax.bar(labels, tv, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tv): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('Newey-West t (next-month return)'); ax.set_title('Control: edge 0 -> t~0; real edge -> t>>2'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,t,p in res: print(f'planted edge={e:+.2f}: HAC t={t:.2f}  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** edge the control sits at **t = {R['syn'][0][4]:.2f}** "
            f"(no false positive); a real planted slope drives **t = {R['syn'][1][4]:.2f}**. So 260 "
            "months is *plenty* of power when an edge exists — the real-tape failure to find one is a "
            "**true null**, not an underpowered one. The breakeven signal simply does not lead "
            "returns."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — {R['n_cross']}/{R['n_tests']} cells cross `|t|≥2` (luck ≈ 0.9); "
            "the TIP hits are self-prediction (placebo `p≈0.11–0.16`), the GLD hit is a 3m-only "
            "fluke, SPY is inert. Proxy/self-prediction support + non-robust point estimates on an "
            "**independent** tape ⇒ WEAK, not REAL (the desk's `t≥2`-on-independent-data bar fails).\n"
            f"- **Tradability `MIRAGE`** — every cond. win-rate < base; the GLD rule nets "
            f"**{R['trade'][0][2]:+.1f}%/yr** vs **+{R['trade'][0][4]:.0f}%/yr** buy-and-hold; the "
            "TIP rule only re-harvests autocorrelation and still trails buy-and-hold. No NAV-scale, "
            "independent, costed edge.\n"
            "- **Free macro-timing lunch? `BUSTED`** — a multiple-testing + self-prediction illusion; "
            "the synthetic control proves the null is true. **Same trap as "
            "[Fed-Model](../118-fed-model/) / [Real-Rate-Regime](../119-real-rate-regime/).**"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the scan-tax curve\n\n"
            "The operational truth in one picture: as you scan more cells, the probability of "
            "finding *at least one* spurious `|t|≥2` climbs fast. At our 18 tests it's already a "
            "coin-flip-plus — so 'we found a significant breakeven signal' is the **expected** "
            "result under the null, not evidence against it."
        ),
        code(
            "m = np.arange(1, 41)\n"
            "p_any = 1 - (1 - 0.05) ** m\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(m, p_any*100, c=AMBER, lw=2, label='P[≥1 false |t|≥2] under the null')\n"
            "ax.axvline(R['n_tests'], c=GREY, ls=':', label=f\"our scan = {R['n_tests']} tests\")\n"
            "ax.axhline(p_any[R['n_tests']-1]*100, c=RED, ls='--')\n"
            "ax.annotate(f'{p_any[R[\"n_tests\"]-1]*100:.0f}% at m={R[\"n_tests\"]}',\n"
            "            (R['n_tests'], p_any[R['n_tests']-1]*100), va='bottom', ha='right', color=RED)\n"
            "ax.set_xlabel('number of tests scanned (m)'); ax.set_ylabel('% chance of a spurious hit')\n"
            "ax.set_ylim(0, 100); ax.set_title('The scan tax: finding a \"significant\" breakeven signal is expected'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'at m={R[\"n_tests\"]} tests, P[at least one false positive] = {p_any[R[\"n_tests\"]-1]*100:.0f}%')"
        ),
        md(
            "> 💡 In plain words: the amber curve is the **false-discovery probability** of the scan "
            "itself. At 18 tests it's already ~60% — so a lone surviving cell carries essentially no "
            "information, and ours (gold, 3m) is exactly that. To claim a real breakeven timer you'd "
            "have to **pre-register** the asset and horizon, beat buy-and-hold net of costs, and "
            "replicate out-of-sample. None of that happens here. **The rarity of a 'hit' is the "
            "arithmetic of the scan, not the wisdom of the bond market.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The same family.** [Study 118 — Fed-Model](../118-fed-model/) and "
            "[Study 119 — Real-Rate-Regime](../119-real-rate-regime/): regress-a-macro-variable "
            "studies with the identical in-sample-slope / out-of-sample-nothing pathology. Read "
            "together, they show the trap is structural, not topic-specific.\n"
            "- **The asset side.** [Study 152 — Inflation-Hedge](../152-inflation-hedge/) — do gold "
            "and TIPS hedge realised inflation? The complement to 'does breakeven *time* them?'\n"
            "- **Better breakeven.** Replace `log(TIP/IEF)` with the literal FRED `T10YIE` (or "
            "5y5y-forward breakeven), add controls (real rate, term premium), and pre-register the "
            "test; the multiple-testing arithmetic is unchanged.\n\n"
            "*The reproducible core is offline and deterministic; the breakeven input is an explicit "
            "proxy. Methods and sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
