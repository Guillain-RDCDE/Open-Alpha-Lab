"""Generate the two narrative notebooks for Study 429 (Mass Index).

    cd notebooks
    python build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY (and panel)
OHLCV under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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
# SPY daily OHLCV (yfinance, auto-adjusted), 2005-01-03 -> 2026-06-23 (21.4y, 5,401 days,
# as-of 2026-06-23, fingerprint aec1923d846c). Mass Index 27/26.5 bulge, 1-day lag.
R = dict(
    asof="2026-06-23", start="2005-01-03", end="2026-06-23", years=21.4, n_days=5401,
    fp="aec1923d846c", n_bulge=36,
    # raw event study: (horizon, bulge_mean, base_mean, win, t0, t_vs_base, placebo_p)
    raw=[(5, 0.0015, 0.0024, 0.64, 0.32, -0.20, 0.757),
         (10, -0.0049, 0.0047, 0.58, -0.70, -1.38, 0.531),
         (20, 0.0007, 0.0094, 0.67, 0.06, -0.76, 0.968),
         (40, 0.0251, 0.0186, 0.69, 2.30, 0.59, 0.269)],
    # fade event study: (horizon, fade_mean, base_mean, win, t0, t_vs_base, placebo_p)
    fade=[(5, 0.0084, -0.0005, 0.53, 1.95, 2.06, 0.039),
          (10, 0.0085, -0.0010, 0.47, 1.25, 1.39, 0.116),
          (20, 0.0159, -0.0010, 0.56, 1.42, 1.50, 0.039),
          (40, 0.0034, -0.0018, 0.53, 0.29, 0.44, 0.761)],
    # timing race (hold 20, long/flat, 1bp): annualised net Sharpe (excess of cash)
    fade_sharpe=0.754, hold_sharpe=0.637, fade_cagr=0.122, hold_cagr=0.108,
    fade_mdd=-0.485, hold_mdd=-0.552, fade_tim=0.90, fade_turn=2.5, t_vs_hold=0.67,
    # variants: (name, sharpe, t_vs_hold)
    variants=[("hold10 L/F", 0.711, 1.01), ("hold10 L/S", 0.745, 1.00),
              ("hold20 L/F", 0.754, 0.67), ("hold20 L/S", 0.735, 0.66),
              ("hold40 L/F", 0.657, -0.82), ("hold40 L/S", 0.461, -0.84)],
    # cost sweep (hold20 L/F): (cost_bps, sharpe, t_vs_hold)
    costs=[(0.0, 0.755, 0.69), (1.0, 0.754, 0.67), (2.0, 0.753, 0.66), (5.0, 0.748, 0.60)],
    # panel: (ticker, n_bulge, fade20_mean, t0, fade_sharpe, hold_sharpe, t_vs_hold)
    panel=[("SPY", 36, 0.0159, 1.42, 0.754, 0.637, 0.67),
           ("QQQ", 29, -0.0062, -0.48, 0.834, 0.768, 0.27),
           ("IWM", 19, 0.0073, 0.34, 0.536, 0.473, 0.88),
           ("DIA", 31, 0.0063, 0.57, 0.738, 0.619, 0.56),
           ("GLD", 23, -0.0124, -0.97, 0.600, 0.648, -1.34)],
    # synthetic control (fade @ h=20; timing L/S hold 25):
    # (edge, detected, fade20_mean, t0, placebo_p, fade_sharpe, hold_sharpe, t_vs_hold)
    syn=[(0.0, 57, -0.0045, -0.50, 0.930, 0.257, 0.364, -0.85),
         (3.0, 79, 0.1668, 3.94, 0.000, 2.787, 0.258, 4.74)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Calls_reversals%3F: Busted](https://img.shields.io/badge/Calls_reversals%3F-Busted-8b949e?style=flat-square)\n\n"
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

from mass_index import data, strategy as st

ASOF = "2026-06-23"
HAVE_REAL = data.have_real("SPY")
if HAVE_REAL:
    SPY = data.load_real("SPY")
    SPY = SPY[SPY.index <= ASOF]      # drop the partial current bar (house rule)
else:
    SPY = None
print("real SPY cache present:", HAVE_REAL,
      "| days:", (0 if SPY is None else len(SPY)),
      "| last:", (None if SPY is None else SPY.index[-1].date()))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Mass Index \"range bulge\" call a reversal? 📐\n"
            "### A 1990s range-expansion indicator, checked against what actually happens next\n\n"
            + BADGES +
            "Here's a piece of charting lore from 1992. A technician named **Donald Dorsey** noticed "
            "that big trend changes are often preceded not by price doing anything special, but by the "
            "daily **range** — the gap between the high and the low — quietly **widening**. He bottled "
            "that into the **Mass Index**: add up, over 25 days, how \"bulgy\" the recent range has "
            "gotten. When the index pushes **above 27** and then slips back **below 26.5**, Dorsey "
            "called it a **reversal bulge** — a warning that the current trend is about to flip.\n\n"
            "It's a lovely story: *the range tells you a turn is coming before the price does.* So we "
            "did the obvious thing — computed the Mass Index on **SPY** over **21 years**, found every "
            "bulge, and asked one blunt question: **after a bulge fires, does the market actually "
            "reverse — or does the next month look like any other month?**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the base-rate contrast and the placebo "
            "test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| How often does the bulge even fire? | **Rarely** — about **{R['n_bulge']}** times in 21 "
            "years (~1.7/yr). It's a rare event, so we have to test it like one. |\n"
            "| After a bulge, does the market reverse? | **No.** The next 5-40 days look like *any* "
            "stretch of tape — the average return after a bulge basically **matches the normal base "
            "rate**. |\n"
            f"| Is there *any* directional edge? | **A whiff, but not enough.** \"Fade the trend after a "
            f"bulge\" is mildly positive, but the best signal (*t* = **{R['fade'][0][4]:.2f}**) falls "
            "short of the bar — and it even points the *wrong* way on the Nasdaq and on gold. |\n"
            "| Can you trade it? | **Not really.** Turned into a timing rule it just **becomes "
            "buy-and-hold** (it's in the market 90% of the time) with no reliable extra return. |\n\n"
            "> The bulge is real — the range really does widen — but a widening range is just "
            "**volatility clustering**, not a crystal ball. \"It calls a reversal\" is **busted** on "
            "the tape."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Reversals are born in the range, not the price. The Mass Index sums the ratio of a "
            "9-day average of the daily high-low range to an average of that average, over 25 days. "
            "When it rises **above 27** and falls back **below 26.5**, the range has bulged and "
            "collapsed — a **reversal bulge** — and a trend change is imminent. Use it with a trend "
            "filter: when the bulge fires, **fade the prevailing trend**.\"*\n\n"
            "Donald Dorsey published the **Mass Index** in *Technical Analysis of Stocks & Commodities* "
            "(June 1992). It's a staple in charting packages (TradingView, StockCharts). The selling "
            "point is the *reversal* call — a warning the trend is tired. So the question is simple: "
            "**does a bulge actually precede a turn, or is it just the market being noisy?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a *range* signal genuinely **forecast reversals**, that would be a real and tradable "
            "crack in efficiency: you'd brace for the turn before the price confirmed it, sidestep the "
            "drawdown, and flip with the new trend. Reversal-timing is the holy grail of technical "
            "analysis precisely because it's so valuable — and so hard.\n\n"
            "But there's a trap. The range widening before a big move is **volatility clustering** — a "
            "well-known fact (turbulent days cluster together) that says *a move is coming* but says "
            "**nothing about which way**. A genuine reversal signal has to beat that: it has to predict "
            "**direction**, not just turbulence. We'll hold the bulge to exactly that bar."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['years']:.0f} years** of daily SPY ({R['start']} → {R['end']}), compute the "
            "Mass Index, and find every reversal bulge. Then:\n\n"
            "1. **What happens next?** For each bulge, measure the **forward 5 / 10 / 20 / 40-day "
            "return** — and compare it to the **base rate** (the average return over *all* the other "
            "days). If the bulge means anything, the after-bulge returns should look different.\n"
            "2. **Which way?** A reversal is *directional*, so we also measure the return **signed "
            "against the prevailing trend** — \"fade the trend.\" A real reversal makes that positive.\n"
            "3. **Could it be luck?** Only ~36 bulges fired. We draw thousands of random sets of 36 "
            "dates and see how often pure chance produces a move this big (the **placebo**).\n\n"
            "What would make us say *mirage*? If the after-bulge return matches the base rate, the "
            "directional version doesn't clear the bar, and a random set of dates does just as well — "
            "three strikes."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, what *is* a bulge?** Here's the Mass Index on SPY with the 27 / 26.5 thresholds, "
            "and a marker on every bar where a reversal bulge completes."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mi = st.mass_index(SPY); bulge = st.bulge_signal(SPY)\n"
            "    bdays = SPY.index[bulge.values]\n"
            "    fig,(a1,a2) = plt.subplots(2,1,figsize=(9.8,6.0),sharex=True,gridspec_kw={'height_ratios':[2,1]})\n"
            "    a1.plot(SPY.index, SPY['close'], c=GREY, lw=1.0)\n"
            "    for d in bdays: a1.axvline(d, c=RED, alpha=.35, lw=1.0)\n"
            "    a1.set_yscale('log'); a1.set_ylabel('SPY close (log)'); a1.set_title(f'SPY with the {len(bdays)} Mass-Index reversal bulges (red)')\n"
            "    a2.plot(mi.index, mi.values, c='black', lw=.9)\n"
            "    a2.axhline(27, c=RED, ls='--', lw=1); a2.axhline(26.5, c=AMBER, ls='--', lw=1)\n"
            "    a2.set_ylabel('Mass Index'); a2.set_ylim(22, 32)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('bulges fired:', int(bulge.sum()))\n"
            "else:\n"
            "    print('no cache; see frozen numbers. bulges =', R['n_bulge'])"
        ),
        md(
            f"About **{R['n_bulge']}** bulges in 21 years — and notice they cluster around turbulent "
            "patches (2008, 2011, 2020), which is the first clue: the bulge is firing on **volatility**, "
            "not on a reliable turning point."
        ),
        md(
            "**Now the money question: what happens after a bulge?** Average forward return after a "
            "bulge (red) vs the everyday base rate (grey), at 5 / 10 / 20 / 40 days. If the bulge "
            "warned of anything, the red bars should stand apart."
        ),
        code(
            "H=[r[0] for r in R['raw']]\n"
            "if HAVE_REAL:\n"
            "    bm=[]; bb=[]\n"
            "    for h in H:\n"
            "        e=st.event_study(SPY,h,fade=False); bm.append(e['bulge_mean']*100); bb.append(e['base_mean']*100)\n"
            "else:\n"
            "    bm=[r[1]*100 for r in R['raw']]; bb=[r[2]*100 for r in R['raw']]\n"
            "x=np.arange(len(H)); fig,ax=plt.subplots(figsize=(9.0,4.4))\n"
            "ax.bar(x-.2, bb, .4, color=GREY, label='base rate (every other day)')\n"
            "ax.bar(x+.2, bm, .4, color=RED, label='after a bulge')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in H]); ax.set_ylabel('avg forward return (%)')\n"
            "ax.set_title('After a bulge looks just like any other stretch'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('after-bulge %:', [round(v,2) for v in bm]); print('base-rate %:', [round(v,2) for v in bb])"
        ),
        md(
            "The red bars **don't stand apart** — if anything they're a touch *lower* than the grey "
            "base rate at 5/10/20 days. The market doesn't reverse after a bulge; it just keeps doing "
            "its usual thing. The lone tall red bar at 40 days is just the market's normal upward drift "
            "showing through (the grey bar is nearly as tall)."
        ),
        md(
            "**But a reversal is directional — maybe it pays to *fade the trend*?** Here's the forward "
            "return signed against the prevailing trend after each bulge. A real reversal makes this "
            "**positive**; the dashed line is where significance (*t* = 2) would start."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fm=[]; tt=[]\n"
            "    for h in H:\n"
            "        e=st.event_study(SPY,h,fade=True); fm.append(e['bulge_mean']*100); tt.append(e['t_vs_zero'])\n"
            "else:\n"
            "    fm=[r[1]*100 for r in R['fade']]; tt=[r[4] for r in R['fade']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "a1.bar([f'{h}d' for h in H], fm, color=[GREEN if v>0 else RED for v in fm])\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('fade return (%)'); a1.set_title('Fade-the-trend return after a bulge')\n"
            "a2.bar([f'{h}d' for h in H], tt, color=[AMBER if abs(v)<2 else GREEN for v in tt])\n"
            "a2.axhline(2,c=GREEN,ls='--',label='t = 2 (the bar)'); a2.axhline(-2,c=RED,ls='--')\n"
            "a2.set_ylabel('one-sample t'); a2.set_title('...but it never clears the bar'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('fade %:', [round(v,2) for v in fm]); print('t:', [round(v,2) for v in tt])"
        ),
        md(
            f"There's a *whiff* of something — the fade is positive at every horizon, and the 5-day "
            f"*t* gets to **{R['fade'][0][4]:.2f}**. But it **never reaches the *t* = 2 bar** the desk "
            "requires, the win-rate is basically a coin flip, and (as we'll see on other markets) it "
            "even points the *wrong way* on the Nasdaq and gold. Suggestive is not the same as real."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** After a bulge, SPY's forward return **matches the base rate** at "
            "every horizon; the directional fade peaks at a sub-significant *t* = "
            f"**{R['fade'][0][4]:.2f}** and flips sign across markets. The bulge doesn't predict a turn.\n"
            "- **Tradability — Mirage.** As a timing rule it's in the market ~90% of the time and just "
            "reproduces buy-and-hold — no reliable extra return.\n"
            "- **\"Calls a reversal\"? — Busted.** A widening range is **volatility clustering**: it "
            "says a move may be coming, never which way."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — it just becomes buy-and-hold\n\n"
            "Suppose you traded the recipe anyway: on a bulge, fade the trend for 20 days, otherwise "
            "stay invested. Growth of \\$1 (excess of cash) vs simply buying and holding."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pos=st.fade_position(SPY,hold=20); bk=st.book_returns(SPY,pos,cost_bps=1.0)\n"
            "    eq_hold=(1+bk['asset_excess']).cumprod(); eq_f=(1+bk['strat_excess']).cumprod(); idx=SPY.index\n"
            "    fs=st.sharpe(bk['strat_excess'].values); hs=st.sharpe(bk['asset_excess'].values)\n"
            "else:\n"
            "    idx=pd.bdate_range('2005-01-03',periods=R['n_days'])\n"
            "    eq_hold=pd.Series(np.linspace(1,np.exp(R['hold_cagr']*R['years']),R['n_days']),index=idx)\n"
            "    eq_f=pd.Series(np.linspace(1,np.exp(R['fade_cagr']*R['years']),R['n_days']),index=idx)\n"
            "    fs=R['fade_sharpe']; hs=R['hold_sharpe']\n"
            "fig,ax=plt.subplots(figsize=(9.6,4.5))\n"
            "ax.plot(idx,eq_hold,c=GREY,lw=2,label=f'buy & hold (Sharpe {hs:.2f})')\n"
            "ax.plot(idx,eq_f,c=RED,lw=1.6,label=f'post-bulge fade (Sharpe {fs:.2f})')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (excess, log)')\n"
            "ax.set_title('The fade rule basically traces buy-and-hold'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'fade Sharpe {fs:.2f}  vs  buy-hold {hs:.2f}  | HAC t of the difference = {R[\"t_vs_hold\"]:.2f}')"
        ),
        md(
            f"The red line **hugs** the grey one. Yes, the fade Sharpe (**{R['fade_sharpe']:.2f}**) is a "
            f"hair above buy-and-hold (**{R['hold_sharpe']:.2f}**) — but the statistical test of the "
            f"difference (HAC *t* = **{R['t_vs_hold']:.2f}**) says that gap is **noise**. The rule "
            "trades ~2.5x/yr and sits invested 90% of the time: it *is* buy-and-hold, give or take an "
            "occasional defensive month that happened to help over this sample. **Mirage.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Other thresholds.** We used Dorsey's 27 / 26.5. You could tune them until something "
            "looks good in-sample — but that's curve-fitting (see the desk's data-mining demos), and "
            "the *standard* recipe is the honest test.\n"
            "- **Other markets.** On a 5-ETF panel (SPY/QQQ/IWM/DIA/GLD) the fade points the *wrong "
            "way* on QQQ and GLD — a real reversal signal would agree across markets.\n"
            "- **The honest lesson.** A *range* indicator measures **volatility**, and volatility "
            "clusters. \"The range will tell you which way price turns\" mixes up two different "
            "things — that a move is coming, and which direction it goes. The first is true; the "
            "second is where the bulge goes quiet. Think you can squeeze a real reversal signal out of "
            "it? Fork it and show the green bar above *t* = 2."
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
            "# The Mass Index reversal bulge — a quantitative teardown 🔬\n"
            "### Bulge event study (forward return vs base rate, raw & trend-signed) · one-sample & "
            "Welch *t* + a rare-event placebo · a net-Sharpe fade race with a HAC *t* on the daily "
            "difference · cost/variant sweeps · a 5-ETF panel · a synthetic planted-reversal control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Dorsey's "
            "Mass Index is sold as a **reversal** detector built from the **range**. We test that "
            "literally: treat each bulge as an event, measure the forward return against the base "
            "rate (raw *and* signed against the trend), then turn it into a timing rule and race it "
            "**net of costs, excess-of-cash** against buy-and-hold.\n\n"
            "> ⚠️ **Data note.** SPY daily OHLCV (yfinance, auto-adjusted = **total-return** proxy), "
            f"{R['start']}→{R['end']}, **{R['years']:.0f} years**, as-of **{R['asof']}** (the partial "
            "2026-06-24 bar is dropped). Offline core + synthetic control are deterministic. Methods "
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
            f"| **Signal** | `NONE` | Forward return after {R['n_bulge']} bulges matches the base rate "
            f"(20-day bulge **{R['raw'][2][1]*100:+.2f}%** vs base **{R['raw'][2][2]*100:+.2f}%**, "
            f"*t* vs base **{R['raw'][2][5]:+.2f}**); the directional fade peaks at one-sample *t* = "
            f"**{R['fade'][0][4]:.2f}** — below the *t ≥ 2* bar — and flips sign across the panel. |\n"
            f"| **Tradability** | `MIRAGE` | Post-bulge fade net Sharpe **{R['fade_sharpe']:.2f}** vs "
            f"buy-and-hold **{R['hold_sharpe']:.2f}**, but HAC *t* of the daily difference = "
            f"**{R['t_vs_hold']:.2f}** (insignificant); long {R['fade_tim']*100:.0f}% of the time, "
            "~2.5 round-trips/yr — it *is* buy-and-hold. |\n"
            f"| **Calls reversals?** | `BUSTED` | A bulge is volatility clustering; the synthetic "
            f"control with a *planted* reversal lights up (*t* = **{R['syn'][1][3]:.2f}**), proving the "
            "harness would catch a real one — and the real tape shows none. |\n\n"
            "> 💡 In plain words: a *reversal* signal must predict **direction**, not just turbulence. "
            "The bulge predicts turbulence (range widening) and nothing about which way price goes."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Dorsey's Mass Index is a rolling sum of a single/double EMA ratio of the daily range:\n\n"
            "$$\\mathrm{MI}_t = \\sum_{i=0}^{24} \\frac{\\mathrm{EMA}_9(H-L)_{t-i}}"
            "{\\mathrm{EMA}_9\\!\\big(\\mathrm{EMA}_9(H-L)\\big)_{t-i}}.$$\n\n"
            "The **reversal bulge** completes on the first bar where $\\mathrm{MI}$, having risen above "
            "**27**, falls back below **26.5**. Let $B$ be the set of those bars and $r_{t\\to t+H}$ the "
            "forward $H$-day return entered one day later. The believer's hypotheses:\n\n"
            "- **H1 (something happens).** $\\mathbb{E}[r_{t\\to t+H}\\mid t\\in B]$ differs from the "
            "unconditional base rate.\n"
            "- **H2 (it reverses).** Signed against the prevailing trend $T_t$, "
            "$\\mathbb{E}[-T_t\\, r_{t\\to t+H}\\mid t\\in B] > 0$ with $t \\ge 2$.\n"
            "- **H3 (it's tradable).** A post-bulge fade rule beats buy-and-hold net of costs "
            "(HAC *t* of the daily difference $> 2$).\n\n"
            f"We find **H1 rejected** (after-bulge ≈ base rate, *t* vs base ≤ +0.59), **H2 rejected** "
            f"(best one-sample *t* = {R['fade'][0][4]:.2f}, sign-unstable across the panel), **H3 "
            f"rejected** (HAC *t* = {R['t_vs_hold']:.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the base rate is the whole game\n\n"
            "Two honesty rules govern this test. **(a) Always contrast with the base rate.** A "
            "one-sample *t* of after-bulge returns against **zero** can be \"passed\" by mere market "
            "drift — over 40 days SPY drifts up ~1.9% on *any* random window. The honest statistic is "
            "the **Welch *t* of bulge vs non-bulge** (does the bulge sample differ from everyday "
            "tape?). **(b) Treat it as the rare event it is.** Only ~36 bulges fired in 21 years, so "
            "we draw thousands of random equal-size trigger sets and ask how often chance reproduces "
            "the observed |mean| (the placebo). For the tradable race we book **excess-of-cash vs "
            "excess-of-cash**, apply **one execution lag once** (signal at close $t$, return over "
            "$t+1$), charge **one-way costs × NAV turnover**, and judge by the **HAC (Newey-West) *t*** "
            "of the daily difference."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY daily OHLCV (yfinance, auto-adjusted), {R['start']}→{R['end']}, "
            f"**{R['n_days']:,}** sessions / {R['years']:.0f}y; 5-ETF panel for robustness.\n"
            "- **Indicator.** Mass Index (EMA 9, sum window 25); bulge = MI > 27 then < 26.5.\n"
            "- **Event study.** Forward 5/10/20/40-day return after each bulge, raw and **signed "
            "against the 25-day-SMA trend**, vs the base rate.\n"
            "- **Inference.** One-sample *t* vs 0; **Welch *t* vs the non-bulge base**; a 5,000-draw "
            "random-trigger placebo (rare-event null).\n"
            "- **Timing.** Post-bulge fade, hold 20d; signal at close $t$, position over $t+1$; costs "
            "1 bp one-way × NAV; borrow 50 bps/yr on shorts.\n"
            "- **Race.** Net annualised Sharpe (excess-of-cash) vs buy-and-hold; **HAC *t*** of the "
            "daily difference; variant & cost sweeps.\n"
            "- **Positive control.** A synthetic panel with a *planted* post-bulge reversal (knob "
            "`edge`): `edge=0` must NOT manufacture significance; a real `edge` must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The bulge event study — after-bulge return vs base rate\n\n"
            "Raw forward return after each bulge (red) vs the non-bulge base rate (grey), with the "
            "Welch *t* of the difference. If the bulge carried information, the red bars separate."
        ),
        code(
            "H=[r[0] for r in R['raw']]\n"
            "if HAVE_REAL:\n"
            "    bm=[];bb=[];tvb=[]\n"
            "    for h in H:\n"
            "        e=st.event_study(SPY,h,fade=False); bm.append(e['bulge_mean']*100); bb.append(e['base_mean']*100); tvb.append(e['t_vs_base'])\n"
            "else:\n"
            "    bm=[r[1]*100 for r in R['raw']]; bb=[r[2]*100 for r in R['raw']]; tvb=[r[5] for r in R['raw']]\n"
            "x=np.arange(len(H)); fig,(a1,a2)=plt.subplots(1,2,figsize=(11.0,4.3))\n"
            "a1.bar(x-.2,bb,.4,color=GREY,label='base rate'); a1.bar(x+.2,bm,.4,color=RED,label='after bulge')\n"
            "a1.set_xticks(x);a1.set_xticklabels([f'{h}d' for h in H]);a1.set_ylabel('mean forward return (%)')\n"
            "a1.set_title('After-bulge return tracks the base rate'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in H], tvb, color=[GREEN if abs(v)>=2 else AMBER for v in tvb])\n"
            "a2.axhline(2,c=GREEN,ls='--'); a2.axhline(-2,c=RED,ls='--'); a2.set_ylabel('Welch t (bulge - base)')\n"
            "a2.set_title('...and the difference is never significant'); plt.tight_layout(); plt.show()\n"
            "print('bulge%:',[round(v,2) for v in bm]); print('base%:',[round(v,2) for v in bb]); print('t_vs_base:',[round(v,2) for v in tvb])"
        ),
        md(
            f"> 💡 In plain words: the after-bulge return is **statistically indistinguishable** from a "
            f"random day — the Welch *t* never leaves the ±2 band (it's *negative* at 5/10/20d). The "
            f"one tall 40-day bar ({R['raw'][3][1]*100:+.2f}%) is just market drift (base "
            f"{R['raw'][3][2]*100:+.2f}%, *t* vs base only {R['raw'][3][5]:+.2f})."
        ),
        md(
            "### 4b · The directional test — does fading the trend pay?\n\n"
            "Forward return **signed against the prevailing trend** after a bulge (a real reversal → "
            "positive), with the one-sample *t* and the rare-event placebo *p*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fm=[];tt=[];pp=[]\n"
            "    for h in H:\n"
            "        e=st.event_study(SPY,h,fade=True); pl=st.placebo_pvalue(SPY,h,fade=True,n_draws=5000)\n"
            "        fm.append(e['bulge_mean']*100); tt.append(e['t_vs_zero']); pp.append(pl['p_value'])\n"
            "else:\n"
            "    fm=[r[1]*100 for r in R['fade']]; tt=[r[4] for r in R['fade']]; pp=[r[6] for r in R['fade']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.0,4.3))\n"
            "a1.bar([f'{h}d' for h in H], tt, color=[GREEN if abs(v)>=2 else AMBER for v in tt])\n"
            "a1.axhline(2,c=GREEN,ls='--',label='t = 2 bar'); a1.axhline(-2,c=RED,ls='--')\n"
            "a1.set_ylabel('one-sample t (fade vs 0)'); a1.set_title('Fade signal: a whiff, never the bar'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in H], pp, color=GREY); a2.axhline(.05,c=RED,ls='--',label='p = 0.05')\n"
            "a2.set_ylabel('placebo p (random triggers)'); a2.set_title('Rare-event placebo'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('fade%:',[round(v,2) for v in fm]); print('t0:',[round(v,2) for v in tt]); print('placebo_p:',[round(v,3) for v in pp])"
        ),
        md(
            f"> 💡 In plain words: the fade is positive everywhere, and 5d/20d brush *p* = 0.039 — but "
            f"the one-sample *t* tops out at **{R['fade'][0][4]:.2f}**, below the desk's *t ≥ 2* bar, "
            "and the pattern is non-monotone (gone by 40 days). On 36 events, this is the texture of "
            "*almost-nothing*, and the inference bar forbids calling it real. **WEAK at most → NONE.**"
        ),
        md(
            "### 4c · The timing race — net Sharpe & the HAC *t*\n\n"
            "Post-bulge fade (hold 20d, long/flat) vs buy-and-hold, growth of \\$1 excess-of-cash, and "
            "the HAC *t* of the daily (fade − buy-hold) difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pos=st.fade_position(SPY,hold=20); bk=st.book_returns(SPY,pos,cost_bps=1.0)\n"
            "    eq_hold=(1+bk['asset_excess']).cumprod(); eq_f=(1+bk['strat_excess']).cumprod(); idx=SPY.index\n"
            "    fs=st.sharpe(bk['strat_excess'].values); hs=st.sharpe(bk['asset_excess'].values)\n"
            "    tval=st.hac_t(bk['strat_excess'].values-bk['asset_excess'].values)\n"
            "else:\n"
            "    idx=pd.bdate_range('2005-01-03',periods=R['n_days'])\n"
            "    eq_hold=pd.Series(np.linspace(1,np.exp(R['hold_cagr']*R['years']),R['n_days']),index=idx)\n"
            "    eq_f=pd.Series(np.linspace(1,np.exp(R['fade_cagr']*R['years']),R['n_days']),index=idx)\n"
            "    fs=R['fade_sharpe']; hs=R['hold_sharpe']; tval=R['t_vs_hold']\n"
            "fig,ax=plt.subplots(figsize=(9.6,4.5))\n"
            "ax.plot(idx,eq_hold,c=GREY,lw=2,label=f'buy & hold ({hs:.2f})')\n"
            "ax.plot(idx,eq_f,c=RED,lw=1.6,label=f'post-bulge fade ({fs:.2f})')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (excess, log)')\n"
            "ax.set_title(f'Fade hugs buy-and-hold (HAC t of the difference = {tval:.2f})'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'fade Sharpe {fs:.3f}  buy-hold {hs:.3f}  HAC t = {tval:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the fade Sharpe (**{R['fade_sharpe']:.2f}**) edges buy-and-hold "
            f"(**{R['hold_sharpe']:.2f}**), but HAC *t* = **{R['t_vs_hold']:.2f}** — the gap is noise. "
            f"At {R['fade_tim']*100:.0f}% time-in-market and ~2.5 round-trips/yr, the rule *is* "
            "buy-and-hold with an occasional defensive tilt."
        ),
        md(
            "### 4d · Variants, costs & the panel — nothing significant, the sign isn't stable\n\n"
            "Left: hold-length / long-short variants (none clear the bar; 40-day *hurts*). Right: the "
            "panel — the fade flips sign on QQQ and GLD."
        ),
        code(
            "if HAVE_REAL:\n"
            "    var=[]\n"
            "    for hold,ls,nm in [(10,False,'h10 L/F'),(20,False,'h20 L/F'),(20,True,'h20 L/S'),(40,False,'h40 L/F')]:\n"
            "        rr=st.run_experiment(SPY,hold=hold,long_short=ls,cost_bps=1.0); var.append((nm,rr['t_vs_hold']))\n"
            "    pan=[]\n"
            "    for t,_,_,_,_,_,_ in R['panel']:\n"
            "        if data.have_real(t):\n"
            "            bb=data.load_real(t); bb=bb[bb.index<=ASOF]; e=st.event_study(bb,20,fade=True); pan.append((t,e['bulge_mean']*100))\n"
            "        else:\n"
            "            row=[p for p in R['panel'] if p[0]==t][0]; pan.append((t,row[2]*100))\n"
            "else:\n"
            "    var=[('h10 L/F',R['variants'][0][2]),('h20 L/F',R['variants'][2][2]),('h20 L/S',R['variants'][3][2]),('h40 L/F',R['variants'][4][2])]\n"
            "    pan=[(p[0],p[2]*100) for p in R['panel']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.0,4.3))\n"
            "a1.bar([v[0] for v in var],[v[1] for v in var],color=[GREEN if abs(v[1])>=2 else AMBER for v in var])\n"
            "a1.axhline(2,c=GREEN,ls='--'); a1.axhline(-2,c=RED,ls='--'); a1.set_ylabel('HAC t vs buy-hold'); a1.set_title('No variant clears the bar')\n"
            "a2.bar([p[0] for p in pan],[p[1] for p in pan],color=[GREEN if p[1]>0 else RED for p in pan])\n"
            "a2.axhline(0,c='k',lw=.8); a2.set_ylabel('fade@20 mean (%)'); a2.set_title('Fade flips sign across markets')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('variant HAC t:',[(n,round(t,2)) for n,t in var]); print('panel fade@20%:',[(t,round(v,2)) for t,v in pan])"
        ),
        md(
            "> 💡 In plain words: every timing variant's HAC *t* sits inside ±2 (the 40-day fade is "
            "*negative* — it sits out up-legs), and the event-study fade is **positive on SPY/IWM/DIA "
            "but negative on QQQ and GLD**. A real effect points the same way everywhere; a sign-flip "
            "across markets is the fingerprint of small-sample noise."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic synthetic panel with a **planted** post-bulge reversal (knob `edge`): "
            "`edge=0` must NOT manufacture significance; a real `edge` must light up. Both hold — so "
            "the dead real-tape result is a true negative, not a broken harness."
        ),
        code(
            "res=[]\n"
            "for edge in (0.0, 3.0):\n"
            "    bars,truth=data.synthetic_panel(n_days=6000,edge=edge,seed=429)\n"
            "    e=st.event_study(bars,20,fade=True); pl=st.placebo_pvalue(bars,20,fade=True,n_draws=3000)\n"
            "    res.append((edge,e['bulge_mean']*100,e['t_vs_zero'],pl['p_value']))\n"
            "labels=[f'edge={e:.0f}\\n(real reversal)' if e>0 else f'edge={e:.0f}\\n(no reversal)' for e,_,_,_ in res]\n"
            "tv=[r[2] for r in res]\n"
            "fig,ax=plt.subplots(figsize=(8.4,4.3))\n"
            "ax.bar(labels,tv,color=[GREY,GREEN],width=.5); ax.axhline(2,c=GREEN,ls='--',label='t = 2 bar')\n"
            "for i,t in enumerate(tv): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('one-sample t (fade @ 20d)'); ax.set_title('Control: no reversal -> t~0; planted reversal -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,m,t,p in res: print(f'edge={e:.1f}: fade@20={m:+.2f}% t0={t:+.2f} placebo_p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted reversal the fade sits at *t* = "
            f"**{R['syn'][0][3]:.2f}** (placebo *p* = {R['syn'][0][4]:.2f} — no false positive); with a "
            f"**real** planted reversal it explodes to *t* = **{R['syn'][1][3]:.2f}** (placebo *p* = "
            f"{R['syn'][1][4]:.3f}, timing Sharpe {R['syn'][1][5]:.2f} vs {R['syn'][1][6]:.2f}). The "
            "engine *can* bank a post-bulge reversal when it exists — and on the real SPY tape, it "
            "doesn't exist."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — after-bulge return matches the base rate at every horizon (20-day "
            f"*t* vs base **{R['raw'][2][5]:+.2f}**; the 40-day one-sample *t* = {R['raw'][3][4]:.2f} is "
            "calendar drift); the directional fade peaks at *t* = "
            f"**{R['fade'][0][4]:.2f}** (sub-2) and flips sign on QQQ/GLD. Literature says reversal; "
            "this tape can't certify it.\n"
            f"- **Tradability `MIRAGE`** — post-bulge fade Sharpe **{R['fade_sharpe']:.2f}** vs "
            f"buy-and-hold **{R['hold_sharpe']:.2f}**, HAC *t* = **{R['t_vs_hold']:.2f}** "
            f"(insignificant); long {R['fade_tim']*100:.0f}% of the time, ~2.5 round-trips/yr. It is "
            "buy-and-hold.\n"
            "- **Calls reversals? `BUSTED`** — the bulge is volatility clustering; the synthetic "
            f"control with a planted reversal lights up (*t* = **{R['syn'][1][3]:.2f}**) while the real "
            "tape stays flat. Range tells you *a* move may come, never which way."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — no separable edge to deploy\n\n"
            "The cost sweep makes the point: because the rule barely trades, costs are nearly "
            "irrelevant — which only underlines that there's no edge for costs to eat. Net Sharpe vs "
            "one-way cost, against the buy-and-hold line."
        ),
        code(
            "cb=[c[0] for c in R['costs']]\n"
            "if HAVE_REAL:\n"
            "    shp=[st.run_experiment(SPY,hold=20,cost_bps=c)['fade_sharpe'] for c in cb]\n"
            "else:\n"
            "    shp=[c[1] for c in R['costs']]\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(cb,shp,'o-',c=RED,lw=2,label='post-bulge fade')\n"
            "ax.axhline(R['hold_sharpe'],c=GREY,ls='--',lw=2,label=f'buy & hold ({R[\"hold_sharpe\"]:.2f})')\n"
            "for x,y in zip(cb,shp): ax.annotate(f'{y:.2f}',(x,y),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net Sharpe (excess of cash)')\n"
            "ax.set_title('Costs barely move it — there is no edge for them to eat'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cost sweep:',{c:round(s,3) for c,s in zip(cb,shp)})"
        ),
        md(
            f"> 💡 In plain words: from 0 to 5 bps the fade Sharpe moves from "
            f"**{R['costs'][0][1]:.2f}** to **{R['costs'][3][1]:.2f}** — almost flat, because it trades "
            "so little. The thing that's missing isn't cheaper execution; it's a signal. **MIRAGE.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Threshold search is a trap.** Grid-searching the 27 / 26.5 levels (or the EMA/sum "
            "windows) until something beats buy-and-hold is curve-fitting (see the desk's data-mining "
            "demos). The *standard* recipe is the honest test, and it fails.\n"
            "- **Volatility vs direction.** The Mass Index is a clean **volatility-clustering** "
            "detector — useful for sizing risk, not for calling turns. Pairing it with a genuine "
            "directional signal is a different study.\n"
            "- **Why a range can't name the direction.** $H-L$ is symmetric in up- and down-moves; a "
            "function of the range carries magnitude, not sign. \"The range tells you which way price "
            "reverses\" is a category error — which is exactly what the flat tape, the sign-flipping "
            "panel, and the lit-up synthetic control jointly show.\n\n"
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
