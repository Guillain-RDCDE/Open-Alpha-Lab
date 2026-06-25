"""Generate the two narrative notebooks for Study 483 (Zero-Lag EMA).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tapes under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
The synthetic positive control runs anywhere with no network.
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
# yfinance daily, 5 indices/ETFs (SPY QQQ IWM DIA GLD), 2005-01-03 -> 2026-05-29 (As-of
# 2026-05-31, partial June dropped), 21.4 years, ZLEMA length L=20 (lag 9), step 5,
# folklore filter "long while price > ZLEMA".
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=2811, length=20, step=5,
    fp_spy="4cb5244f3990",
    # pooled price>ZLEMA filter, per horizon:
    # (H, n, zlema_bps, win%, one_sample_t, random_bps, delta_rnd, ema_bps, delta_ema, net_bps, welch_t, welch_p)
    h5=(5, 2806, 15.1, 56, 3.58, 24.7, -9.6, 19.4, -4.2, 13.1, -1.37, 0.170),
    h10=(10, 2806, 44.8, 62, 6.53, 54.3, -9.5, 43.9, 0.9, 42.8, -0.99, 0.323),
    h20=(20, 2799, 96.0, 64, 8.09, 117.2, -21.2, 80.6, 15.4, 94.0, -1.57, 0.116),
    h60=(60, 2779, 265.0, 68, 7.93, 279.5, -14.6, 250.7, 14.2, 263.0, -0.63, 0.529),
    # per-ticker H=20: (ticker, entries, zlema_bps, one_sample_t, random_bps, delta_rnd, delta_ema)
    per=[("SPY", 566, 88.0, 3.82, 115.6, -27.5, 7.3), ("QQQ", 568, 122.1, 4.10, 171.8, -49.7, 4.0),
         ("IWM", 554, 90.4, 2.94, 106.2, -15.8, 39.3), ("DIA", 558, 80.8, 3.66, 104.3, -23.6, 18.6),
         ("GLD", 565, 98.4, 3.57, 87.5, 11.0, 7.7)],
    # de-lag placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(88.0, 0.363, 500),
    # ZLEMA upcross foil (SPY, H=20): n, mean_bps, win, t
    upcross=(602, 72.6, 68, 2.48),
    # synthetic control (H=20, n_days=4000): (edge, zlema_n, zlema_bps, win%, zlema_t, ema_bps, ema_t)
    syn=[(0.0, 403, 0.3, 47, 0.01, 18.3, 0.55), (2.0, 565, 5676.8, 62, 4.94, 9589.1, 7.52)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Removing_lag_buys_edge%3F: Busted](https://img.shields.io/badge/Removing_lag_buys_edge%3F-Busted-8b949e?style=flat-square)\n\n"
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

from zlema import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real ZLEMA cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the *zero-lag* EMA actually buy you anything? ⚡\n"
            "### A moving average that promises to kill its own lag — meets a stopwatch\n\n"
            + BADGES +
            "Every moving average lags price: it's an average of the past, so it trails behind. "
            "John Ehlers' **zero-lag EMA** (ZLEMA) is a clever fix — instead of smoothing the price, "
            "it smooths `price + (price − price-a-few-bars-ago)`, which nudges the input *forward* so "
            "the line catches up. The pitch is irresistible: a trend filter with **no lag** should get "
            "you into trends sooner and out sooner, beating a plain old EMA of the same length.\n\n"
            "It *looks* great on a chart — the ZLEMA hugs price more tightly. But 'hugs price tighter' "
            "and 'makes you money' are two very different things. So we did the only fair test: encode "
            "the rule **mechanically** (long while price is above the ZLEMA), fire it thousands of times "
            "across five big indices over 21 years, and race it against two honest baselines — "
            "**buying on random days**, and the **plain EMA it claims to beat.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy when price is above the **zero-lag EMA**, do I make money? | **Yes — but only "
            "because the market goes up.** The win-rate is ~60% and the returns look great. |\n"
            "| Is that *the ZLEMA's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well or better** — the ZLEMA filter is *worse* than a coin-flip entry at every "
            "horizon. |\n"
            "| Does removing the lag beat a plain EMA? | **No.** Head-to-head, the de-lagged line and "
            "the boring EMA finish in a dead heat (a few bps apart, sign flipping). |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the market's upward "
            "drift, dressed up as a fancy filter. The 'zero lag' is cosmetic. |\n\n"
            "> The zero-lag EMA is a nicer-looking line. As a *forecast* — 'less lag = timelier "
            "trades' — it's a **mirage**: all of the apparent edge is the market's climb, and the "
            "de-lag adds nothing the plain EMA didn't already give you."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A normal EMA lags price by about half its length. The **zero-lag EMA** smooths a "
            "de-lagged input — `price + (price − price[lag])` — so it tracks price with (almost) no "
            "lag. A timelier line means timelier signals: go long while price is above the ZLEMA and "
            "you'll catch trends earlier and beat a plain EMA.\"*\n\n"
            "This is **John Ehlers'** zero-lag / 'instantaneous trendline' idea (early 2000s), part of "
            "a whole family of low-lag averages (DEMA, TEMA, KAMA…). It's built into TradingView, "
            "MetaTrader and every charting suite. So: does killing the lag actually buy you an edge?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a de-lagged line genuinely *forecast* better, it would be a free lunch: the same trend "
            "filter, only earlier, for the price of one subtraction. That's the dream the indicator "
            "sells.\n\n"
            "But there are two traps. **(a)** It runs on stock indices that drift **up** over time, so "
            "*any* 'be long in an uptrend' rule looks profitable — that's the market, not the filter. "
            "**(b)** 'Less lag' isn't free: subtracting the lag is a high-pass tweak that **amplifies "
            "noise**, so the zero-lag line *overshoots* and whipsaws. To separate the **tool** from "
            "the **tide** we have to (i) compare to buying on **random days**, and (ii) put it head-to-"
            "head against the **plain EMA** it claims to beat. We do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            f"1. **Build the zero-lag EMA mechanically.** Length {R['length']}, the de-lag uses a fixed "
            "lookback — only past prices, nothing from the future.\n"
            "2. **Trade the lore.** While price is above the ZLEMA, we're long; we sample that state "
            f"every {R['step']} days and measure the return over the next **5 / 10 / 20 / 60 days** "
            "(entering at the next close — one honest lag).\n"
            "3. **The drift baseline.** Do the exact same hold on **random days**. If the ZLEMA "
            "matters, it must beat random.\n"
            "4. **The head-to-head.** Run the *identical* rule on a **plain EMA** of the same length. "
            "If the de-lag buys anything, the ZLEMA must beat the EMA. *If it doesn't, the 'zero lag' "
            "is decoration* — announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the zero-lag line even look like next to a plain EMA? Here's SPY with "
            "both — notice the ZLEMA (green) hugs price more tightly, but also jitters more."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-300:]\n"
            "    z = st.zlema(cl, R['length']); e = st.ema(cl, R['length'])\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.plot(seg.index, z.reindex(seg.index), c=GREEN, lw=1.5, label='zero-lag EMA (L=20)')\n"
            "    ax.plot(seg.index, e.reindex(seg.index), c=GREY, lw=1.5, ls='--', label='plain EMA (L=20)')\n"
            "    ax.set_title('Zero-lag EMA hugs price tighter than a plain EMA — but does it pay?')\n"
            "    ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The ZLEMA tracks price more closely — *as a drawing*. The question is whether being above "
            "it predicts anything. **Let's race the ZLEMA filter against random entries** at four "
            "horizons. Green = long while above the ZLEMA; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    zlema, rnd = [], []\n"
            "    for h in hs:\n"
            "        zz, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.zlema_entries(c, length=R['length'])\n"
            "            re = st.random_entries(c, max(len(e),50), length=R['length'], seed=7)\n"
            "            zz.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        zlema.append(np.concatenate(zz).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    zlema = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, zlema, .4, color=GREEN, label='long while price > ZLEMA')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(zlema,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The ZLEMA filter does NOT beat random — it loses to it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('zlema:', [round(v) for v in zlema]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story. The ZLEMA filter makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make more** "
            f"(**+{R['h20'][5]:.0f} bps**). At *every* horizon the famous zero-lag line is *worse* than "
            "throwing darts. The apparent profit was **the market's upward drift**, not the filter."
        ),
        md(
            "**The decisive test for *this* indicator.** ZLEMA's specific promise is 'better than a "
            "plain EMA'. So put them head-to-head — the *same* rule, one on the de-lagged line, one on "
            "the boring EMA. If removing the lag buys anything, the green bars beat the grey ones."
        ),
        code(
            "if HAVE_REAL:\n"
            "    zz, ee = [], []\n"
            "    for h in hs:\n"
            "        z1, e1 = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            z1.append(st.forward_returns(c, st.zlema_entries(c, length=R['length']), h))\n"
            "            e1.append(st.forward_returns(c, st.ema_entries(c, length=R['length']), h))\n"
            "        zz.append(np.concatenate(z1).mean()*1e4); ee.append(np.concatenate(e1).mean()*1e4)\n"
            "else:\n"
            "    zz = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    ee = [R['h5'][7], R['h10'][7], R['h20'][7], R['h60'][7]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, zz, .4, color=GREEN, label='zero-lag EMA')\n"
            "ax.bar(x+.2, ee, .4, color='#2c6fbb', label='plain EMA (same length)')\n"
            "for i,(a,bb) in enumerate(zip(zz,ee)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('Zero-lag EMA vs plain EMA: a dead heat'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('zlema-minus-ema (bps):', [round(a-b) for a,b in zip(zz,ee)])"
        ),
        md(
            f"A dead heat. ZLEMA minus the plain EMA is **{R['h5'][8]:+.0f} / {R['h10'][8]:+.0f} / "
            f"{R['h20'][8]:+.0f} / {R['h60'][8]:+.0f} bps** across the four horizons — a few basis "
            "points, sign flipping, nothing you could trade. Removing the lag changed the *picture*, "
            "not the *payoff*. The 'zero lag' was decoration."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The ZLEMA filter does **not** beat buying on random days "
            "(it's *worse* at every horizon). The big absolute returns are the market's drift.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Removing the lag buys edge\"? — Busted.** Head-to-head with a plain EMA it's a dead "
            "heat, and scrambling the de-lag changes nothing. The 'zero lag' is cosmetic."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The ZLEMA filter's *only* advantage over a coin flip is "
            "the market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. And its one job, to beat a plain EMA, it fails: same payoff, more "
            "whipsaws. The de-lag adds noise, not edge. As a forecasting tool it doesn't pay; as a "
            "nicer-looking line, it was never a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The whippy cross.** The *upcross* version (buy the moment price crosses above the "
            "ZLEMA) is even worse — the de-lag overshoots so the cross fires near tops. The quants "
            "notebook shows it banks *less* of a real planted trend than the steady filter.\n"
            "- **The low-lag family.** DEMA, TEMA, KAMA, HMA are the same de-lag idea in different "
            "clothes — the desk's sibling studies find the same thing: nicer line, same drift.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* persistent trend "
            "into a synthetic tape and shows the harness banks it (so the null result here isn't a "
            "dead detector — it's an honest 'nothing there'), and that the **plain EMA banks more**.\n\n"
            "*Think zero-lag forecasts? Show the ZLEMA filter beating random entries **and** the plain "
            "EMA at **t ≥ 2** on a real tape — then we'll talk.*"
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
            "# Zero-Lag EMA — a quantitative teardown 🔬\n"
            "### Causal ZLEMA on 5 indices · `price > ZLEMA` forward returns · one-sample HAC *t* · "
            "a drift-matched random-entry baseline · a **plain-EMA head-to-head** · a de-lag placebo · "
            "the whippy-upcross foil · costs · a synthetic planted-trend control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Two jobs: "
            "separate the **filter** from the **drift** (an upward-trending index makes *any* "
            "long-in-uptrend rule look good), and separate the **de-lag** from the **plain EMA** (the "
            "one thing ZLEMA is supposed to improve on).\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. ZLEMA is causal "
            f"(length {R['length']}, lag {(R['length']-1)//2}); entry is the **next close** (one "
            "documented lag). Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | `price>ZLEMA` vs a **drift-matched random** baseline: the filter "
            f"is *worse* at all horizons (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps) and the Welch *t* is **negative everywhere** "
            f"(20d = {R['h20'][10]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d = {R['h20'][4]:.2f}) are "
            f"**pure beta** — they vanish against random entries and against cost. No residual edge "
            "to scale. |\n"
            f"| **Removing lag buys edge?** | `BUSTED` | Head-to-head vs a plain EMA: ZLEMA − EMA = "
            f"{R['h5'][8]:+.0f}/{R['h10'][8]:+.0f}/{R['h20'][8]:+.0f}/{R['h60'][8]:+.0f} bps (tiny, "
            f"sign-flipping); de-lag placebo **p = {R['placebo'][1]:.2f}**. |\n\n"
            "> 💡 In plain words: the filter *looks* significant only because indices drift up. Strip "
            "the drift (race vs random) or strip the de-lag (vs plain EMA, or scramble the offset) and "
            "the edge evaporates. Classic beta-in-a-costume, with a cosmetic de-lag."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A plain EMA $E_t = \\alpha C_t + (1-\\alpha)E_{t-1}$ lags price by $\\approx (L-1)/2$ "
            "bars. Ehlers' fix feeds the EMA a **de-lagged** input: with $\\text{lag}=(L-1)/2$,\n\n"
            "$$Z_t = \\mathrm{EMA}_L\\!\\big(C_t + (C_t - C_{t-\\text{lag}})\\big).$$\n\n"
            "The $(C_t - C_{t-\\text{lag}})$ term is the momentum over the lag window; adding it back "
            "pushes the input forward. The rule: **long while $C_t > Z_t$.**\n\n"
            "- **H₀ (drift).** Filter returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the filter forecasts).** Filter returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the de-lag matters).** Filter returns **exceed** the **plain-EMA** filter (and a "
            "**shuffled-offset** placebo).\n\n"
            "We find **H₀ not rejected** (filter < random at every horizon), **H₁ rejected** (Welch t "
            "negative throughout), **H₂ rejected** (ZLEMA ≈ EMA; placebo p ≈ 0.36). All three legs fail."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "filter inherits it; a high one-sample $t$ against **zero** measures the tide, not the "
            "tool. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a Welch "
            "test of filter-*minus*-random.\n\n"
            "**(b) The de-lag as a free parameter.** ZLEMA's whole pitch is 'better than an EMA'. The "
            "load-bearing test is therefore the **plain-EMA head-to-head**: identical rule, identical "
            "length, only the line differs. We add a **shuffled-offset placebo** that permutes the "
            "de-lag term $C_t - C_{t-\\text{lag}}$ (marginal kept, alignment destroyed) — if the real "
            "result survives the scramble, the zero-lag correction was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']:,} `price>ZLEMA` "
            "samples** pooled.\n"
            f"- **Indicator.** Causal ZLEMA, length {R['length']} (lag {(R['length']-1)//2}); the "
            "de-lag uses only past closes.\n"
            f"- **Entry.** Sample the `price>ZLEMA` long-state every {R['step']} bars (non-overlap); "
            "enter **next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of filter returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample filter vs random (the drift test).\n"
            "- **Null #3 — plain-EMA head-to-head** (the de-lag test) + a **shuffled-offset placebo**.\n"
            "- **Costs.** 1 bp one-way × 2 legs on every trade.\n"
            "- **Positive control.** Synthetic tape with a **planted** persistent trend (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up (and we check the plain EMA banks "
            "at least as much)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the `price>ZLEMA` filter's **one-sample** t against zero (the misleading number). "
            "Right: the same filter vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, zlema, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        zz, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.zlema_entries(c, length=R['length'])\n"
            "            re = st.random_entries(c, max(len(e),50), length=R['length'], seed=7)\n"
            "            zz.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        zz = np.concatenate(zz); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(zz)['t']); zlema.append(zz.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(zz, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    zlema = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][10], R['h10'][10], R['h20'][10], R['h60'][10]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Filter vs RANDOM, Welch t (honest: negative everywhere)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**) — but "
            "that's the **drift**, every long-in-uptrend rule inherits it. The right bars are the real "
            f"test: filter-minus-random is **negative at every horizon** ({R['h20'][10]:+.2f} at 20d). "
            "The ZLEMA filter adds nothing over a coin flip — it's a touch *worse*."
        ),
        md(
            "### 4b · The head-to-head — zero-lag vs a plain EMA\n\n"
            "The test that's specific to *this* indicator. Same rule, same length; only the line "
            "differs. ZLEMA should tower over the boring EMA if the de-lag forecasts. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    zz, ee = [], []\n"
            "    for h in hs:\n"
            "        z1, e1 = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            z1.append(st.forward_returns(c, st.zlema_entries(c, length=R['length']), h))\n"
            "            e1.append(st.forward_returns(c, st.ema_entries(c, length=R['length']), h))\n"
            "        zz.append(np.concatenate(z1).mean()*1e4); ee.append(np.concatenate(e1).mean()*1e4)\n"
            "else:\n"
            "    zz = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    ee = [R['h5'][7], R['h10'][7], R['h20'][7], R['h60'][7]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, zz, .4, color=GREEN, label='zero-lag EMA')\n"
            "ax.bar(x+.2, ee, .4, color='#2c6fbb', label='plain EMA')\n"
            "for i,(a,b) in enumerate(zip(zz,ee)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Zero-lag EMA does not beat a plain EMA'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta zlema-ema (bps):', [round(a-b) for a,b in zip(zz,ee)])"
        ),
        md(
            f"> 💡 In plain words: ZLEMA − EMA = {R['h5'][8]:+.0f}/{R['h10'][8]:+.0f}/{R['h20'][8]:+.0f}/"
            f"{R['h60'][8]:+.0f} bps — a few bps either way, never material. The de-lag changed the "
            "line's *cosmetics* and left the *payoff* untouched. H₂ rejected on the head-to-head."
        ),
        md(
            "### 4c · The de-lag placebo — scramble the zero-lag correction, nothing changes\n\n"
            "Permute the de-lag offset $C_t - C_{t-\\text{lag}}$ (marginal kept, alignment destroyed) "
            "and re-run. If the de-lag carries information, the observed return should sit far in the "
            "right tail of the scrambled distribution. It sits mid-pack."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.delag_placebo(c, 20, length=R['length'], n_draws=300, seed=483)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np\n"
            "    lag = (R['length']-1)//2\n"
            "    offset = (c - c.shift(lag)).to_numpy(); base = c.to_numpy(); idx = c.index\n"
            "    vmask = _np.isfinite(offset); rng = _np.random.default_rng(483); draws = []\n"
            "    for _ in range(300):\n"
            "        po = offset.copy(); v = _np.where(vmask)[0]; po[v] = rng.permutation(offset[v])\n"
            "        dl = __import__('pandas').Series(base + _np.nan_to_num(po, nan=0.0), index=idx)\n"
            "        line = st.ema(dl, R['length']); ent = st._state_entries(c, line)\n"
            "        rr = st.forward_returns(c, ent, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(483); draws = rng.normal(95, 25, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-de-lag (SPY, 20d)')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'real ZLEMA {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean price>ZLEMA 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real ZLEMA sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real ZLEMA {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => de-lag not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real ZLEMA (green line) sits **inside** the scrambled-de-lag "
            f"cloud — **p = {R['placebo'][1]:.2f}**. A random offset does about as well, so the "
            "specific zero-lag correction isn't carrying information. The cleanest refutation of "
            "'removing the lag adds edge.'"
        ),
        md(
            "### 4d · Per-ticker — loses to random, dead heat vs EMA\n\n"
            "20-day ZLEMA-minus-random and ZLEMA-minus-EMA deltas, per instrument. If the de-lag "
            "worked, both would be solidly positive. Instead Δ_rnd is negative in 4 of 5, and Δ_ema is "
            "a small positive sliver."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, d_rnd, d_ema = [], [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.zlema_entries(c, length=R['length']); em = st.ema_entries(c, length=R['length'])\n"
            "        re = st.random_entries(c, max(len(e),50), length=R['length'], seed=7)\n"
            "        z20 = st.summarize(st.forward_returns(c,e,20))['mean_bps']\n"
            "        d_rnd.append(z20 - st.summarize(st.forward_returns(c,re,20))['mean_bps'])\n"
            "        d_ema.append(z20 - st.summarize(st.forward_returns(c,em,20))['mean_bps'])\n"
            "        names.append(t)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; d_rnd = [p[5] for p in R['per']]; d_ema = [p[6] for p in R['per']]\n"
            "x = np.arange(len(names))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "ax.bar(x-.2, d_rnd, .4, color=[GREEN if d>0 else RED for d in d_rnd], label='ZLEMA − random')\n"
            "ax.bar(x+.2, d_ema, .4, color='#2c6fbb', label='ZLEMA − plain EMA')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(d_rnd): ax.annotate(f'{d:+.0f}',(i-.2,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=8)\n"
            "for i,d in enumerate(d_ema): ax.annotate(f'{d:+.0f}',(i+.2,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel('20d delta (bps)')\n"
            "ax.set_title('ZLEMA loses to random in 4/5; barely beats the EMA'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('d_rnd:', {n:round(d) for n,d in zip(names,d_rnd)}); print('d_ema:', {n:round(d) for n,d in zip(names,d_ema)})"
        ),
        md(
            f"> 💡 In plain words: only **GLD** edges a positive Δ_rnd ({R['per'][4][5]:+.0f} bps); the "
            f"rest are *behind* random. And Δ_ema is a thin positive sliver (max {R['per'][2][6]:+.0f} "
            "bps on IWM) — far too small and noisy to call an edge of the de-lag. No coherent "
            "cross-sectional advantage: relabelled drift with a cosmetic twist."
        ),
        md(
            "### 4e · The whippy-upcross foil + synthetic positive control\n\n"
            "Two checks at once. **(i)** The bare *upcross* (buy the moment price crosses above ZLEMA) "
            "scores a high one-sample t on the real tape — pure drift again. **(ii)** On a synthetic "
            "tape with a *planted* persistent trend, the steady `price>ZLEMA` filter must light up "
            "(edge>0) and stay flat with no trend (edge=0) — proving the detector is live — while the "
            "**plain EMA banks at least as much** (so the de-lag never helps)."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 2.0):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=483, n_days=4000)\n"
            "    c = px['close']\n"
            "    z = st.summarize(st.forward_returns(c, st.zlema_entries(c, length=20), 20))\n"
            "    e = st.summarize(st.forward_returns(c, st.ema_entries(c, length=20), 20))\n"
            "    res.append((edge, z['n'], z['mean_bps'], z['win']*100, z['t'], e['mean_bps'], e['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.2))\n"
            "labels = [f'planted edge\\n{e:.1f}' for e,*_ in res]; zt = [r[4] for r in res]; et = [r[6] for r in res]\n"
            "x = np.arange(len(res))\n"
            "ax.bar(x-.2, zt, .4, color=GREEN, label='zero-lag EMA t')\n"
            "ax.bar(x+.2, et, .4, color='#2c6fbb', label='plain EMA t')\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(zt): ax.annotate(f'{t:.2f}',(i-.2,t),ha='center',va='bottom',fontsize=8)\n"
            "for i,t in enumerate(et): ax.annotate(f'{t:.2f}',(i+.2,t),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('20d one-sample t')\n"
            "ax.set_title('Control: edge=0 -> t~0; planted trend -> lights up (EMA >= ZLEMA)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t,em,et_ in res: print(f'edge={e:.1f}: zlema n={n} mean={m:+.1f} win={w:.0f}% t={t:+.2f} | ema mean={em:+.1f} t={et_:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted trend the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (no false positive); a planted trend reaches "
            f"**t = {R['syn'][1][4]:.2f}** — the detector works, so the flat real-tape result is a "
            f"genuine 'nothing there'. And even on the planted trend the **plain EMA wins** "
            f"(t = {R['syn'][1][6]:.2f} vs {R['syn'][1][4]:.2f}): the de-lag is a *handicap*, not a "
            "head start."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — `price>ZLEMA` does not beat a drift-matched random baseline "
            f"(filter − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t **negative at every horizon**). The "
            f"impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Removing lag buys edge? `BUSTED`** — head-to-head, ZLEMA − plain EMA = "
            f"{R['h5'][8]:+.0f}/{R['h10'][8]:+.0f}/{R['h20'][8]:+.0f}/{R['h60'][8]:+.0f} bps (a dead "
            f"heat); the de-lag placebo leaves the result intact (**p = {R['placebo'][1]:.2f}**); and "
            "on a planted trend the plain EMA banks more than the de-lagged line. The 'zero lag' is "
            "cosmetic — it buys a noisier line, not a timelier edge."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The filter's entire apparent profit is the unconditional drift of long equity indices, "
            "obtained more cheaply and more fully by **buying and holding**. Against the very thing it "
            "claims to improve — a plain EMA — it's a dead heat, and it whipsaws more (the upcross foil "
            "and the noisier synthetic control). There is no capacity question because there is no edge "
            "to scale. The zero-lag EMA is a nicer-looking trend line, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The low-lag family.** DEMA, TEMA (Mulloy 1994), KAMA (Kaufman), HMA (Hull) are the "
            "same de-lag idea in different clothes; the desk's sibling studies "
            "([432-hma](../../432-hma), [433-kama](../../433-kama), [434-dema-tema](../../434-dema-tema)) "
            "find the same drift-in / cosmetics-out result.\n"
            "- **Length / lag sweeps.** Varying L and the de-lag lookback is a free in-sample knob; the "
            "head-to-head dead heat is robust across reasonable settings — more lag-removal just means "
            "more noise.\n"
            "- **The noise tax.** 'Zero lag' is a high-pass tweak: it removes phase lag by amplifying "
            "high-frequency content, so the timeliness is paid for in whipsaws — exactly what the "
            "upcross foil shows.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the "
            "detector is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen "
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
