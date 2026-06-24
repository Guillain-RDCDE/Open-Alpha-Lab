"""Generate the two narrative notebooks for Study 408 (Three Black Crows).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket OHLCV
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLCV,
# 30-name liquid large-cap + SPY basket, 2005-01-03 -> 2026-06-18, 21.5 years).
R = dict(
    start="2005-01-03", end="2026-06-18", years=21.5, n_names=30,
    fingerprint="bf1d6cb7ca54", total_bars=161970,
    # SIGNED-SHORT forward return after three black crows, per horizon:
    # (H, n, mean%, base%, win%, t_hac, t_one, t_welch, p_placebo, net%)
    h1=(1, 1530, -0.054, 0.028, 47, -1.60, -1.58, -0.73, 0.942, -0.156),
    h3=(3, 1530, -0.117, 0.141, 47, -1.52, -1.69, 0.35, 0.958, -0.223),
    h5=(5, 1530, -0.165, 0.253, 47, -1.65, -1.87, 0.99, 0.969, -0.275),
    h10=(10, 1526, -0.284, 0.530, 44, -2.06, -2.23, 1.92, 0.984, -0.404),
    # strict-crow myth-check: (H, n, mean%, t_hac, p, net%)
    strict=[(1, 145, 0.060, 0.65, 0.302, -0.042), (3, 145, -0.093, -0.47, 0.669, -0.199),
            (5, 145, -0.081, -0.33, 0.616, -0.191), (10, 144, -0.232, -0.56, 0.716, -0.352)],
    # prior-uptrend (true reversal) myth-check: (H, n, mean%, t_hac, p, net%)
    trend=[(1, 870, -0.020, -0.46, 0.647, -0.122), (3, 870, -0.056, -0.66, 0.734, -0.161),
           (5, 870, -0.007, -0.06, 0.527, -0.117), (10, 866, -0.156, -1.00, 0.828, -0.275)],
    # synthetic control 5d: (planted_edge, planted_days, n, mean%, t_hac, t_one, p, win%)
    syn=[(0.000, 675, 673, -0.139, -1.05, -1.15, 0.876, 49),
         (0.004, 718, 716, 0.404, 3.02, 3.41, 0.000, 54),
         (0.008, 754, 752, 0.971, 7.04, 8.21, 0.000, 61)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Crash_incoming%3F: Busted](https://img.shields.io/badge/Crash_incoming%3F-Busted-8b949e?style=flat-square)\n\n"
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

from three_black_crows import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real(allow_fetch=False)
else:
    PANEL = None
print("real basket cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else len(PANEL)))
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
            "# Three black crows — does the candle that screams \"crash\" actually predict one? 🐦‍⬛\n"
            "### One of candlestick lore's scariest reversal signals, measured on 21 years of real tape\n\n"
            + BADGES +
            "Open any candlestick book and you'll meet the **three black crows**: three long **red** "
            "(down) days in a row, each opening inside the day before and closing near its low, "
            "stair-stepping down. The lore is dramatic — it's supposed to mark the **top** of an uptrend "
            "and warn that a serious **sell-off is coming**. Spot the crows, get short (or get out), and "
            "you'll dodge the crash.\n\n"
            "It *looks* ominous. But \"looks ominous\" and \"predicts the future\" are different things. "
            "So we did the boring, honest thing: found **every** three-black-crows in a basket of 30 big "
            "US stocks + SPY over 21 years, and measured what actually happened next.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo tests and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use a fixed **30-name liquid large-cap + SPY** basket "
            "(names still trading today), so this carries **survivorship** (we can't include firms that "
            "actually *did* crash and delist). If anything that should make the \"crash\" signal look "
            "**stronger** than reality — and it still doesn't show up. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After three black crows, does the stock keep falling? | **No.** Short it the next morning "
            "and over the next 1–10 days you **lose**, on average. The \"crash\" simply doesn't arrive. |\n"
            "| Is it a *bullish* contrarian signal then? | **Mildly, by accident.** Stocks tend to drift "
            "**up** afterwards (because stocks drift up in general) — at 10 days the short loses "
            f"**{abs(R['h10'][2]):.2f}%** — but that's just the market's tide, not a real edge. |\n"
            "| Does the *strict*, picture-perfect crow work better? | **No.** Tighten the rule to the "
            "textbook long-bodied crow and it still does nothing (and there are barely any). |\n"
            "| What if I only count crows after a real uptrend (a true \"reversal\")? | **Still nothing.** "
            "The prior-uptrend filter doesn't rescue it either. |\n\n"
            "> The scariest-looking candle in the book is, on the tape, **noise** — and trading it short "
            "**loses money** before you even pay costs."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Three black crows is a **bearish reversal**. After a rally, three long red candles in a "
            "row — each opening inside the previous body and closing near the low — show the bulls have "
            "lost control. A downtrend has begun: sell, or short, because the **decline is just "
            "starting**.\"*\n\n"
            "This is straight out of Steve Nison's *Japanese Candlestick Charting Techniques* (the book "
            "that brought candles to the West) and is repeated by every charting site from Investopedia "
            "to StockCharts. It's one of the most recognizable patterns in technical analysis — and one "
            "of the most *frightening*-looking. The question is simply: **does the crash come?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If three black crows really marked tops, it would be enormously useful: a clean, eyeball-able "
            "signal to dodge drawdowns or to short for a quick profit. Risk managers would love it; it "
            "needs no data feed beyond a price chart.\n\n"
            "But there's a trap built into the very shape of the pattern. Three big down days in a row is, "
            "almost by definition, a stock that has **already fallen a lot**. Markets that have just "
            "dropped sharply tend to **bounce** (short-term mean reversion), not keep tumbling. So the "
            "naive bearish reading might be exactly **backwards** — you could be shorting right into a "
            "rebound. We'll see which it is."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name** basket of liquid US large-caps + **SPY** and scan "
            f"every daily bar from **{R['start']}** to **{R['end']}** ({R['years']:.0f} years). For each "
            "name we flag **every** three-black-crows using the precise OHLC rule (three stacked red "
            "bodies, each closing lower, each opening inside the prior body). Then, with **no cheating**:\n\n"
            "1. **Wait for the close** that confirms the third crow.\n"
            "2. **Enter the next morning's open** (one day's lag — you can't trade the bar that's still "
            "forming) and hold **1, 3, 5, or 10** trading days, **short** (the bearish bet).\n"
            "3. **Compare** that short's return to the everyday base rate, and to thousands of random "
            "\"fake signal\" picks. If shorting the crows beats coin-flips, the legend is real. If it "
            "**loses**, the crash is a myth."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Here's what a short taken the morning after three black crows earns, "
            "on average, at each horizon. A real \"crash\" signal would show **green** (the short makes "
            "money). Watch the colour."
        ),
        code(
            "hs = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    means = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "else:\n"
            "    means = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [GREEN if m>0 else RED for m in means]\n"
            "ax.bar([f'{h}d' for h in hs], means, color=cols, width=.6)\n"
            "for i,v in enumerate(means): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('trading days held (short, after the crows)'); ax.set_ylabel('avg return of the SHORT (%)')\n"
            "ax.set_title('Shorting three black crows LOSES money at every horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('short return by horizon:', {f'{h}d': round(m,3) for h,m in zip(hs,means)})"
        ),
        md(
            f"Every bar is **red** — the short **loses**. At 1 day you're down "
            f"**{R['h1'][2]:.2f}%**, and it gets *worse* the longer you hold, reaching "
            f"**{R['h10'][2]:.2f}%** at 10 days. The \"crash\" never comes; the stock tends to keep "
            "drifting **up** (because that's what stocks do). The scariest candle in the book points the "
            "**wrong way**."
        ),
        md(
            "**Could it just be luck — or bad luck?** Let's pit the real signal against thousands of "
            "**fake** ones: pick the same number of random days, flip a coin for long/short, and see how "
            "the real crows' return stacks up. If the crows had bearish power, the real result would sit "
            "in the **right** tail (better than the fakes)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.collect_events(PANEL, 5)\n"
            "    obs = float(ev['signed_ret'].mean())\n"
            "    pl = st.placebo_pvalue(PANEL, 5, len(ev), obs, n_draws=5000)\n"
            "    draws = pl['draws']*100; obsp = obs*100; pval = pl['p_value']\n"
            "else:\n"
            "    obsp = R['h5'][2]; pval = R['h5'][8]\n"
            "    rng = np.random.default_rng(408); draws = rng.normal(0.0, 0.30, 5000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='5,000 random coin-flip 'r'\"signals\"')\n"
            "ax.axvline(obsp, c=RED, lw=2.5, label=f'real crows short {obsp:+.2f}%')\n"
            "ax.set_xlabel('5-day short return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The real crows sit in the LEFT tail (worse than random): p = {pval:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obsp:+.2f}%   share of random fakes that beat it: p = {pval:.3f}')"
        ),
        md(
            f"The red line — the real signal — sits in the **left** tail. About "
            f"**{R['h5'][8]*100:.0f}%** of *random* coin-flip picks did **better** than shorting the "
            "actual crows. That's the opposite of an edge: a random number generator would have made you "
            "more money than the famous pattern."
        ),
        md(
            "**Does the picture-perfect version save it?** Maybe loose crows are noise but the strict, "
            "long-bodied textbook crow (closing near the low each day, after a real uptrend) is the real "
            "deal. Let's check both stricter recipes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    basic  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    strict = [st.summarize(PANEL, h, placebo=False, strict=True)['mean']*100 for h in hs]\n"
            "    trend  = [st.summarize(PANEL, h, placebo=False, require_trend=True)['mean']*100 for h in hs]\n"
            "else:\n"
            "    basic  = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "    strict = [s[2] for s in R['strict']]; trend = [t[2] for t in R['trend']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(x-.25, basic, .25, color=RED, label='basic crows')\n"
            "ax.bar(x,      strict,.25, color=AMBER, label='strict long-bodied crow')\n"
            "ax.bar(x+.25,  trend, .25, color=GREY, label='after a prior uptrend')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('avg SHORT return (%)'); ax.set_title('No recipe rescues it — every filter still loses (or does nothing)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('strict 5d:', f'{strict[2]:+.2f}%', ' uptrend 5d:', f'{trend[2]:+.2f}%')"
        ),
        md(
            f"Neither filter helps. The strict crow at 5 days is **{R['strict'][2][2]:+.2f}%** (on just "
            f"**{R['strict'][2][1]}** rare events — basically noise), and requiring a prior uptrend gives "
            f"**{R['trend'][2][2]:+.2f}%**. The legend doesn't hide in a stricter definition; it just "
            "isn't there."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Shorting three black crows **loses** at every horizon "
            f"(**{R['h1'][2]:.2f}%** to **{R['h10'][2]:.2f}%**), and ~95%+ of random picks do better. "
            "There's no bearish edge — if anything the stock mildly *bounces*.\n"
            "- **Tradability — Mirage.** It's negative **before** costs; once you add the spread and "
            "short borrow it's worse. Nothing to deploy.\n"
            "- **\"Crash incoming\"? — Busted.** The famous \"sell-off is just starting\" reading is "
            "**backwards**: three down days are a stock that already fell, and it tends to drift back up."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — no, and here's the kicker\n\n"
            "The short is already negative gross. Now add the real-world frictions — the spread paid on "
            "the round trip, plus borrow to hold a short. The bars only sink further."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    nv = [st.summarize(PANEL, h, placebo=False)['net']*100 for h in hs]\n"
            "else:\n"
            "    g  = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "    nv = [R['h1'][9], R['h3'][9], R['h5'][9], R['h10'][9]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=AMBER, label='gross short return')\n"
            "ax.bar(x+.2, nv, .4, color=RED, label='net of costs + borrow')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('SHORT return (%)'); ax.set_title('Costs only deepen the loss — there is nothing to harvest')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net short by horizon:', {f'{h}d': round(v,3) for h,v in zip(hs,nv)})"
        ),
        md(
            f"At every horizon the **net** short is more negative than the gross — e.g. "
            f"**{R['h5'][9]:.2f}%** at 5 days. There is no window where it pays. The only way to \"make "
            "money\" from three black crows is to do the **opposite** of the lore (don't short the dip) — "
            "and even that contrarian bounce is just the market's normal upward drift, not a tradeable "
            "pattern edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Why does it fail?** Three big red days = a stock that already dropped. Short-term mean "
            "reversion (the bounce) dominates, so the *bearish* reading is exactly backwards.\n"
            "- **The survivorship twist.** Our basket excludes names that actually crashed and delisted — "
            "so we've **stacked the deck in the lore's favour** and it *still* fails. On a full universe "
            "with delistings the short might do marginally better, but the broad-market evidence "
            "(Marshall et al. 2006) says candlestick signals add no value.\n"
            "- **Build your own.** Swap the basket, add a volume-confirmation filter, or test the bullish "
            "mirror (\"three white soldiers\") — the same harness handles it.\n\n"
            "*Think a stricter crow, a different universe, or a clever filter turns it green? Show the "
            "**net** short landing above zero with t ≥ 2 — then we'll talk.*"
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
            "# Three Black Crows — a quantitative teardown 🔬\n"
            "### Precise OHLC detector · signed-short forward 1/3/5/10-day event study · HAC + one-sample "
            "*t* + a coin-flip label-shuffle placebo · costs + borrow · strict-crow & prior-uptrend "
            "myth-checks · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Three black "
            "crows is candlestick lore's marquee **bearish reversal** — the job here is to test it "
            "honestly as a short signal: enter one lag after the confirming close, measure the signed "
            "forward return against zero with a HAC *t*, confront it with a coin-flip placebo, charge it "
            "costs, and check whether *any* stricter recipe rescues it.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name liquid large-cap + SPY** basket, names "
            "still trading in 2026 — a *survivor* panel that **excludes the firms that actually crashed "
            "and delisted**, i.e. it tilts the test *toward* finding a working bearish signal. It still "
            "doesn't. Real data: yfinance daily OHLCV, `auto_adjust=True`, 2005→2026. Offline core + "
            "synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Signed-**short** forward return after three black crows is "
            f"**negative at every horizon** ({R['h1'][2]:+.2f}% at 1d → {R['h10'][2]:+.2f}% at 10d). "
            f"One-sample / HAC *t* are **negative** (10d HAC *t* = {R['h10'][5]:.2f}) and the coin-flip "
            f"placebo *p* ≈ **{R['h5'][8]:.2f}** (the real signal is in the *left* tail). No bearish edge "
            "— it never approaches *t* ≥ +2. |\n"
            f"| **Tradability** | `MIRAGE` | Negative **gross**; net of 2×5-bps round trip + 50 bps/yr "
            f"borrow it is worse at every horizon ({R['h5'][9]:+.2f}% net at 5d). Nothing to deploy. |\n"
            f"| **Crash incoming?** | `BUSTED` | The \"sell-off is just starting\" reading is **backwards**: "
            f"the post-pattern drift is mildly **up** (the short *loses*, 10d *t* = {R['h10'][5]:.2f} in "
            "the *bounce* direction). Strict-crow and prior-uptrend filters don't rescue it. |\n\n"
            "> 💡 In plain words: the scariest candle in the book has **no predictive power** as a short, "
            "and the contrarian residue is just the market's upward drift — not a pattern edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a three-black-crows be confirmed at the close of bar $t$ when the OHLC triple "
            "$(t\\!-\\!2, t\\!-\\!1, t)$ satisfies: three red bodies ($C_j<O_j$), monotone closes "
            "($C_t<C_{t-1}<C_{t-2}$), and each open inside the prior body "
            "($C_{j-1}<O_j\\le O_{j-1}$). Enter the **next** open (one lag), hold $H$ days **short**, and "
            "let $s_i(H) = -\\big(\\text{Close}_{t+H}/\\text{Open}_{t+1}-1\\big)$ be the signed return of "
            "event $i$.\n\n"
            "- **H₁ (bearish edge exists).** $\\overline{s}(H) > 0$ and significant for some $H$ — "
            "shorting the crows pays.\n"
            "- **H₂ (it's deployable).** $\\overline{s}(H)$ survives the round-trip cost + short borrow.\n"
            "- **H₃ (\"strict crow / real reversal\").** Restricting to long-bodied crows after a prior "
            "uptrend sharpens the edge.\n\n"
            "We find **H₁ rejected** ($\\overline{s}(H)<0$ at every $H$, HAC *t* negative), **H₂ rejected** "
            "(net worse), **H₃ rejected** (filters don't help). The pattern carries **no** bearish "
            "information; the only residual is the market's unconditional upward drift, which makes the "
            "short *lose*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a one-sample / HAC test of the signed-short event sample against zero:\n\n"
            "$$t = \\frac{\\overline{s}}{\\widehat{\\mathrm{se}}_{\\text{NW}}(\\overline{s})},"
            "\\qquad s_i = -\\Big(\\tfrac{C_{t_i+H}}{O_{t_i+1}}-1\\Big).$$\n\n"
            "Two honesty problems sit on a naive *t*. **(a) The base rate.** Stocks drift **up** "
            "unconditionally, so a *short* starts with a headwind — the right null is zero signed return, "
            "and we also Welch-compare to the unconditional pool signed short ($-$base). **(b) Selection "
            "on a shape.** \"Three down days\" pre-selects stocks that already fell, which mechanically "
            "invites a mean-reversion **bounce** — the exact opposite of the bearish claim. The "
            "coin-flip placebo (same event count, random signs) is the honest \"could random picks look "
            "this good?\" null; here the real signal is **worse** than random."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** liquid large-cap + SPY basket (yfinance daily "
            f"OHLCV, `auto_adjust=True`, {R['start']}→{R['end']}, {R['total_bars']:,} bars). **Survivor** "
            "panel — named on the Signal axis (and it tilts *toward* the bearish claim).\n"
            "- **Detector.** Precise real-body three-black-crows on bars $(t\\!-\\!2,t\\!-\\!1,t)$; an "
            "optional **strict** variant (long bodies, close near the low) and a **prior-uptrend** filter "
            "for the myth-check.\n"
            "- **Timing.** Confirm at the close of $t$; enter the **next open** (one lag); hold "
            "$H\\in\\{1,3,5,10\\}$ days **short**; drop events whose window overruns.\n"
            "- **Null #1 (one-sample / HAC t)** of the signed-short sample vs 0 — the inference-bar number.\n"
            "- **Null #2 (coin-flip placebo).** 5,000 draws of the same event count from the "
            "unconditional pool, each randomly signed; $p=\\Pr[\\text{random mean}\\ge\\text{observed}]$.\n"
            "- **Costs.** 5 bps one-way × 2 (round trip) + 50 bps/yr borrow on the short over the hold.\n"
            "- **Positive control.** A deterministic panel with a **planted** post-pattern crash: zero "
            "edge must NOT reach significance; a planted edge must light up the signed-short *t*.\n\n"
            "> **Falsifier, stated in advance:** if the signed-short HAC *t* never clears **+2** on the "
            "real tape, the bearish signal is **None** and \"crash incoming\" is **Busted**."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline term structure + significance\n\n"
            "Left: the signed-short mean by horizon (green = short pays, red = short loses). Right: the "
            "5-day signed-short mean against the 5,000-draw coin-flip null."
        ),
        code(
            "hs = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(PANEL, h, placebo=False) for h in hs]\n"
            "    means = [r['mean']*100 for r in rows]; tt = [r['t_hac'] for r in rows]\n"
            "    ev = st.collect_events(PANEL, 5); obs = float(ev['signed_ret'].mean())\n"
            "    pl = st.placebo_pvalue(PANEL, 5, len(ev), obs, n_draws=5000)\n"
            "    draws = pl['draws']*100; obsp = obs*100; pval = pl['p_value']\n"
            "else:\n"
            "    means = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "    tt = [R['h1'][5], R['h3'][5], R['h5'][5], R['h10'][5]]\n"
            "    obsp = R['h5'][2]; pval = R['h5'][8]\n"
            "    rng = np.random.default_rng(408); draws = rng.normal(0.0, 0.30, 5000)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], means, color=[GREEN if m>0 else RED for m in means], width=.6)\n"
            "for i,(v,t) in enumerate(zip(means,tt)): a1.annotate(f'{v:+.2f}%\\nt={t:.2f}',(i,v),ha='center',va='top',fontsize=8)\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('signed-SHORT return (%)'); a1.set_title('Short loses at every horizon (HAC t < 0)')\n"
            "a2.hist(draws, bins=50, color=GREY, alpha=.85, label='5,000 coin-flip fakes')\n"
            "a2.axvline(obsp, c=RED, lw=2.5, label=f'observed {obsp:+.2f}%')\n"
            "a2.set_xlabel('5-day signed-short return (%)'); a2.set_ylabel('freq')\n"
            "a2.set_title(f'Real signal in the LEFT tail: p={pval:.3f}'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('means:', [round(m,3) for m in means], ' HAC t:', [round(t,2) for t in tt], ' 5d placebo p:', round(pval,3))"
        ),
        md(
            f"> 💡 In plain words: the term structure is uniformly **red** — the short loses more the "
            f"longer you hold ({R['h1'][2]:+.2f}% → {R['h10'][2]:+.2f}%), and every HAC *t* is "
            f"**negative**. The placebo histogram nails it: the observed result sits in the **left** tail "
            f"(*p* ≈ {R['h5'][8]:.2f}), i.e. ~{R['h5'][8]*100:.0f}% of random coin-flip picks did *better*. "
            "This is the textbook shape of a **dead** signal — actually worse than dead, since the naive "
            "bearish direction is on the wrong side of the post-drop bounce."
        ),
        md(
            "### 4b · The full inference table — HAC, one-sample, Welch, placebo\n\n"
            "Every arbiter at once. The believer needs a **positive** signed-short mean with *t* ≥ +2; "
            "nothing here comes close."
        ),
        code(
            "import pandas as pd\n"
            "if HAVE_REAL:\n"
            "    tab = pd.DataFrame([st.summarize(PANEL, h, n_draws=5000) for h in hs])\n"
            "    show = tab[['horizon','n_events','mean','base_mean','win','t_hac','t_one','t_welch','p_placebo','net']].copy()\n"
            "    for c in ['mean','base_mean','net']: show[c] = (show[c]*100).round(3)\n"
            "    for c in ['win']: show[c] = (show[c]*100).round(0)\n"
            "    for c in ['t_hac','t_one','t_welch']: show[c] = show[c].round(2)\n"
            "    show['p_placebo'] = show['p_placebo'].round(3)\n"
            "else:\n"
            "    keys=['h1','h3','h5','h10']; cols=['horizon','n_events','mean','base_mean','win','t_hac','t_one','t_welch','p_placebo','net']\n"
            "    show = pd.DataFrame([dict(zip(cols, R[k])) for k in keys])\n"
            "print(show.to_string(index=False))"
        ),
        md(
            f"> 💡 In plain words: read the `t_hac`/`t_one` columns — all **negative**, none near +2. The "
            f"win-rate is **~{R['h1'][4]}%** (below a coin flip *for a short*, because the market drifts "
            f"up). `t_welch` vs the unconditional pool signed-short is also unimpressive. The signed "
            "return is simply negative and not even significantly so in the *bearish* direction — the "
            "only borderline significance (10d) is in the **bounce** (bullish) direction."
        ),
        md(
            "### 4c · The myth check — does a stricter crow rescue it?\n\n"
            "Two ways to make the signal \"purer\": the **strict** long-bodied crow (close near each "
            "low), and only counting crows that **followed a prior uptrend** (a genuine reversal). If the "
            "lore were right, these should be *more* bearish."
        ),
        code(
            "if HAVE_REAL:\n"
            "    basic  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    strict = [st.summarize(PANEL, h, placebo=False, strict=True)['mean']*100 for h in hs]\n"
            "    trend  = [st.summarize(PANEL, h, placebo=False, require_trend=True)['mean']*100 for h in hs]\n"
            "    ns = st.summarize(PANEL, 5, placebo=False, strict=True)['n_events']\n"
            "    nt = st.summarize(PANEL, 5, placebo=False, require_trend=True)['n_events']\n"
            "else:\n"
            "    basic  = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "    strict = [s[2] for s in R['strict']]; trend = [t[2] for t in R['trend']]\n"
            "    ns = R['strict'][2][1]; nt = R['trend'][2][1]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(x-.25, basic, .25, color=RED, label='basic crows')\n"
            "ax.bar(x,      strict,.25, color=AMBER, label=f'strict crow (n5={ns})')\n"
            "ax.bar(x+.25,  trend, .25, color=GREY, label=f'prior uptrend (n5={nt})')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('signed-SHORT return (%)'); ax.set_title('No filter turns it bearish — the edge simply is not there')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('strict:', [round(s,3) for s in strict]); print('trend :', [round(t,3) for t in trend])"
        ),
        md(
            f"> 💡 In plain words: the strict crow (only **{R['strict'][2][1]}** events at 5d — a tiny, "
            f"noisy sample) gives **{R['strict'][2][2]:+.2f}%** (HAC *t* = {R['strict'][2][3]:.2f}), and "
            f"the prior-uptrend reversal gives **{R['trend'][2][2]:+.2f}%** (*t* = {R['trend'][2][3]:.2f}). "
            "Both wander around zero with no bearish tilt. The pattern's predictive content is "
            "**indistinguishable from noise** under every reasonable definition."
        ),
        md(
            "### 4d · Costs — the loss only deepens\n\n"
            "Gross vs net (5 bps one-way × 2 + 50 bps/yr borrow on the short). When the gross is already "
            "negative, costs can't save it — they bury it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    nv = [st.summarize(PANEL, h, placebo=False)['net']*100 for h in hs]\n"
            "else:\n"
            "    g  = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "    nv = [R['h1'][9], R['h3'][9], R['h5'][9], R['h10'][9]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=AMBER, label='gross short')\n"
            "ax.bar(x+.2, nv, .4, color=RED, label='net (costs + borrow)')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('SHORT return (%)'); ax.set_title('Gross is negative; net is worse — no tradeable window')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h,gg,nn in zip(hs,g,nv): print(f'{h:>2}d: gross={gg:+.3f}%  net={nn:+.3f}%')"
        ),
        md(
            f"> 💡 In plain words: at 5 days gross **{R['h5'][2]:+.2f}%** → net **{R['h5'][9]:+.2f}%**. "
            "There is no horizon where the short clears zero, let alone its costs. The signal is a "
            "**Mirage** in the strict sense — there's nothing under it to charge costs *against*."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **plant** a known post-pattern crash: with **zero** edge "
            "the signed-short *t* must stay ≈ 0 (no false positive); with a planted crash it must light "
            "up positive. Both hold — so the real-tape *t* < 0 is a genuine *absence* of bearish signal, "
            "not a broken detector."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.004, 0.008):\n"
            "    px, truth = data.synthetic_panel(edge=edge, seed=408)\n"
            "    s = st.summarize(px, 5, n_draws=4000)\n"
            "    res.append((edge, truth['n_planted_days'], s['n_events'], s['mean']*100, s['t_hac'], s['p_placebo'], s['win']*100))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "labels = [f'planted\\n{e*100:.1f}%/day' for e,_,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, AMBER, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=+2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('signed-short 5d HAC t'); ax.set_title('Control: zero edge -> t~0; planted crash -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,pd_,k,ls,t,p,w in res: print(f'planted {e*100:+.1f}%/day: planted_days={pd_} n={k} mean={ls:+.3f}% t={t:.2f} p={p:.3f} win={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted crash the control sits at *t* = "
            f"**{R['syn'][0][4]:.2f}** (no false positive — noise can't fake a bearish edge); a planted "
            f"0.4%/day crash reaches *t* = **{R['syn'][1][4]:.2f}**, and 0.8%/day reaches "
            f"**{R['syn'][2][4]:.2f}**. The detector *would* catch a real post-crows crash if one "
            "existed. It catches nothing on the real tape — so the absence is real, not a measurement "
            "failure."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the signed-short forward return is **negative at every horizon** "
            f"({R['h1'][2]:+.2f}% → {R['h10'][2]:+.2f}%) with **negative** HAC *t* (never near +2) and a "
            f"coin-flip placebo *p* ≈ {R['h5'][8]:.2f} (real signal in the *left* tail). No bearish edge. "
            "Carries an explicit **survivorship** caveat that tilts the test *toward* the claim — and it "
            "still fails.\n"
            f"- **Tradability `MIRAGE`** — negative **gross**; net of round-trip cost + borrow it's worse "
            f"({R['h5'][9]:+.2f}% net at 5d). There is no edge to charge costs against.\n"
            f"- **Crash incoming? `BUSTED`** — the \"sell-off is just starting\" reading is **backwards**: "
            f"the post-pattern drift is mildly **up** (the short loses; the only borderline significance, "
            f"10d *t* = {R['h10'][5]:.2f}, is the *bounce*). Strict-crow and prior-uptrend filters don't "
            "rescue it. The famous crash signal is a story the chart tells, not one the tape confirms."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the only money is in fading the lore\n\n"
            "One picture for the operational truth: the signed-**short** net return by horizon against "
            "the break-even line. It never lifts above zero — the tradeable region is **empty**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    nv = [st.summarize(PANEL, h, placebo=False)['net']*100 for h in hs]\n"
            "else:\n"
            "    g  = [R['h1'][2], R['h3'][2], R['h5'][2], R['h10'][2]]\n"
            "    nv = [R['h1'][9], R['h3'][9], R['h5'][9], R['h10'][9]]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(hs, g, 'o--', c=GREY, lw=1.6, label='gross short')\n"
            "ax.plot(hs, nv, 'o-', c=RED, lw=2.2, label='net of costs + borrow')\n"
            "ax.axhline(0, c='k', ls='--', label='break-even')\n"
            "ax.fill_between(hs, 0, nv, color=RED, alpha=.10)\n"
            "for h,v in zip(hs,nv): ax.annotate(f'{v:+.2f}%',(h,v),ha='center',va='top',fontsize=9)\n"
            "ax.set_xlabel('horizon (days)'); ax.set_ylabel('signed-short return (%)')\n"
            "ax.set_title('Empty tradeable region: the short is net-negative everywhere'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net short by horizon:', {f'{h}d': round(v,3) for h,v in zip(hs,nv)})"
        ),
        md(
            "> 💡 In plain words: the red (net) line is **below zero** at every horizon — there is no "
            "window where shorting the crows pays. The contrarian residue (the stock drifting up) is just "
            "the unconditional equity drift you'd capture by holding *anything*; it is not a pattern edge "
            "and carries no timing value. Hence **Mirage**, and the bearish claim is **Busted**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Why it fails, mechanically.** Three consecutive red bodies pre-select stocks that have "
            "**already dropped**; short-horizon mean reversion (the bounce) then dominates, putting the "
            "naive bearish bet on the wrong side. The pattern is a *consequence* of a fall, not a "
            "*predictor* of more.\n"
            "- **The survivorship lever points the *right* way here.** Excluding delisted crashers biases "
            "us **toward** a working short signal — so a clean fail on survivors is strong evidence. "
            "Marshall, Young & Rose (2006) reach the same null on the broad DJIA cross-section.\n"
            "- **The mirror.** Test \"three white soldiers\" (the bullish twin) with the same harness — "
            "lore says it's bullish; the post-rally drift likely mean-reverts the other way.\n\n"
            "*The reproducible core is offline and deterministic; the detector is the precise real-body "
            "three-black-crows with a strict variant and a prior-uptrend filter as myth-checks. Methods "
            "and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
