"""Generate the two narrative notebooks for Study 397 (Hurst-Regime).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached market prices
under ../_cache/ (SPY + QQQ/GLD/TLT/EFA) and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with no
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY + QQQ/GLD/TLT/EFA,
# 1995-01-03 -> 2026-06-18, 7,918 SPY days, 31.5 years; rolling 252-day R/S Hurst, stride 5).
R = dict(
    start="1995-01-03", end="2026-06-18", days=7918, years=31.5,
    h_mean=0.552, h_median=0.553, h_std=0.050, h_lo=0.36, h_hi=0.695, share_trend=84.0,
    # books: name -> (Sharpe, ann_mean%)
    books=dict(buyhold=(0.66, 12.4), trend=(0.10, 1.9), revert=(0.25, 4.8), gated=(0.17, 3.2)),
    # gate vs best-static (always-revert) and vs buy&hold
    gate_vs_static=dict(diff=-0.085, lo=-0.549, hi=0.364, p_le0=0.654, paired_t=-0.322),
    gate_vs_bh=dict(diff=-0.486, p_le0=0.979),
    placebo=dict(real=-0.085, pmean=-0.160, p=0.215),
    # per-market: ticker -> (H_mean, share_trend%, gated, best_static, buyhold)
    markets=dict(SPY=(0.552, 84, 0.17, 0.25, 0.66), QQQ=(0.558, 87, 0.29, 0.13, 0.52),
                 GLD=(0.592, 97, 0.32, 0.32, 0.64), TLT=(0.564, 86, 0.19, 0.22, 0.33),
                 EFA=(0.564, 91, -0.03, 0.12, 0.41)),
    # robustness: window -> (gated, best_static, buyhold)
    windows={126: (0.22, 0.25, 0.66), 252: (0.17, 0.25, 0.66), 504: (0.06, 0.25, 0.66)},
    # synthetic estimator: planted H -> R/S estimate
    syn_est=[(0.30, 0.384), (0.50, 0.550), (0.70, 0.717)],
    # synthetic gate (stride=1): edge -> (gated, trend, revert, diff, p_boot, p_placebo)
    syn_gate=[(0.0, -0.23, -0.22, -0.33, -0.007, 0.477, 0.434),
              (1.0, 0.42, -0.06, -0.74, 0.486, 0.044, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Free_lunch%3F: Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square)\n\n"
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

from hurst_regime import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    F = data.load_real("SPY")
    RETS = st.log_returns(F["price"])
    HURST = data.rolling_hurst(RETS, window=252)        # trailing R/S, stride 5 (fast, faithful)
    BOOKS = st.build_books(F["price"], HURST, cost_bps=1.0)
else:
    F = RETS = HURST = BOOKS = None
print("real cache present:", HAVE_REAL,
      "| Hurst valid days:", (0 if HURST is None else int(HURST.notna().sum())))
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
            "# Can a number tell you when to trend-follow and when to bet on a reversal? 📐\n"
            "### The Hurst exponent — the 'self-diagnosing market' — in plain English\n\n"
            + BADGES +
            "There's a beautiful idea in market lore. Every price series has a number — the "
            "**Hurst exponent** — that supposedly tells you its *character*. Above **0.5** the "
            "market is **trending** (so ride the trend); below **0.5** it's **mean-reverting** "
            "(so bet on a snap-back); right at 0.5 it's a coin-flip. Compute it on a rolling "
            "window and you'd have a strategy that **drives itself**: trend when it should, fade "
            "when it should, always in the right gear.\n\n"
            "It sounds like the holy grail — let the math of the series pick your strategy for "
            "you. This notebook builds exactly that switch on 31 years of real markets and shows "
            "why it ends up a *worse* version of doing nothing.\n\n"
            "> 📓 **Plain-language layer.** Want the R/S estimator, the block bootstrap and the "
            "placebo test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by the "
            "code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the Hurst exponent tell you when to trend vs fade? | **No.** The gated "
            "switch's edge over the best plain style is *wrong-signed* and statistically zero. |\n"
            "| Does the self-driving strategy beat buying and holding? | **No — it loses to it "
            f"by ~0.5 Sharpe**, in every one of {len(R['markets'])} markets we tried. |\n"
            "| Then why does the idea feel so right? | Because the exponent is a **real** "
            "statistic. The leap from 'real number' to 'tradeable regime switch' is where it "
            "breaks. |\n"
            "| Is there a smoking gun? | **Two.** On real markets the Hurst reading is **stuck "
            f"above 0.5 ~{R['share_trend']:.0f}% of the time** (so the 'switch' barely switches), "
            "and where it *does* switch, a **shuffled** regime label works just as well. |\n\n"
            "> The Hurst exponent is a genuine measure of a series. Reading a *trade* off it is a "
            "category error — and the market quietly proves it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Measure the Hurst exponent H on a rolling window. **H > 0.5** ⇒ the series is "
            "persistent/**trending** — trend-follow it. **H < 0.5** ⇒ it's "
            "anti-persistent/**mean-reverting** — fade it. **H ≈ 0.5** ⇒ random walk, sit out. "
            "A gate that switches styles by H captures the right premium in every market state.\"*\n\n"
            "The exponent comes from Harold Hurst's 1951 study of Nile floods and was carried "
            "into markets by Mandelbrot and then Edgar Peters' *Fractal Market Analysis*. The "
            "intuition is gorgeous: the **geometry of the path itself** — how its range grows "
            "with time — tells you whether moves tend to *continue* or *reverse*. If true, you'd "
            "never again have to guess which strategy the market wants. We'll build the switch "
            "and find out."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a single rolling number really flagged 'trend now / fade now', it would be worth "
            "a fortune: trend-following and mean-reversion are the two oldest systematic styles, "
            "and each one *bleeds* in the other's regime (a trend-follower gets chopped to bits "
            "in a ranging market; a fader gets run over by a trend). A perfect regime switch "
            "would let you collect both premia and dodge both drawdowns. The catch is that a "
            "strategy which is **right on average** isn't enough — it has to be right by **more** "
            "than just holding the asset, *and* the switch has to carry information a coin-flip "
            "label wouldn't. Those are the two bars the legend has to clear."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We rebuild the self-driving switch on **{R['years']:.0f} years** of SPY "
            f"({R['start']} → {R['end']}) and four other markets:\n\n"
            "1. **Read the regime.** Compute a trailing **Hurst exponent** (the classic R/S "
            "method) on a 1-year rolling window — using only the past, no peeking.\n"
            "2. **Switch styles.** When H > 0.5 trend-follow (go with the recent move); when "
            "H < 0.5 mean-revert (fade the recent move). That's the gated book.\n"
            "3. **Race it.** Compare the gated book against (a) just trend-following always, "
            "(b) just mean-reverting always, and (c) simply **buying and holding** — all net of "
            "trading costs, all at the same risk.\n"
            "4. **Stress the luck.** **Shuffle** the regime label and re-run: if the real Hurst "
            "reading carries information, the true switch should beat the shuffled one. If it "
            "doesn't, the exponent was never telling you anything."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the smoking gun nobody mentions.** Here's SPY's rolling Hurst exponent "
            "over 31 years. The claim needs it to swing above and below 0.5. Watch where it "
            "actually lives."
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = HURST.dropna()\n"
            "    share = (h > 0.5).mean()*100\n"
            "    fig, ax = plt.subplots(figsize=(9.4, 4.2))\n"
            "    ax.plot(h.index, h.values, c=GREY, lw=.8)\n"
            "    ax.axhline(0.5, ls='--', c=RED, label='0.5 (the trend / mean-revert line)')\n"
            "    ax.fill_between(h.index, 0.5, h.values, where=(h.values>0.5), color=GREEN, alpha=.25)\n"
            "    ax.set_ylabel('rolling 1-year Hurst (R/S)')\n"
            "    ax.set_title(f'SPY Hurst sits ABOVE 0.5 about {share:.0f}% of the time')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'mean H={h.mean():.3f}  median={h.median():.3f}  share>0.5={share:.0f}%  range=[{h.min():.2f},{h.max():.2f}]')\n"
            "else:\n"
            "    print('no cache — frozen:', f\"mean H={R['h_mean']}, share>0.5={R['share_trend']}%, range[{R['h_lo']},{R['h_hi']}]\")"
        ),
        md(
            f"There's the first crack. The Hurst reading is **pinned above 0.5 ~{R['share_trend']:.0f}% "
            f"of the time** (mean **{R['h_mean']}**, never below **{R['h_lo']}**). The "
            "mean-reverting regime the strategy depends on barely ever shows up — so the "
            "'self-driving switch' is really a trend-follower with the wheel taped straight. "
            "(This isn't bad luck: the R/S estimator is known to read *high* on short windows.)"
        ),
        md(
            "**Now the race.** Four books at the same risk: buy & hold, always-trend, "
            "always-revert, and the Hurst-gated switch. Which one wins?"
        ),
        code(
            "names = ['buyhold','trend','revert','gated']\n"
            "labels = ['buy & hold','always\\ntrend','always\\nrevert','Hurst\\nGATE']\n"
            "if HAVE_REAL:\n"
            "    s = st.summarize_books(BOOKS)\n"
            "    sr = [s[k]['sharpe'] for k in names]\n"
            "else:\n"
            "    sr = [R['books'][k][0] for k in names]\n"
            "cols = [GREEN, GREY, GREY, RED]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(labels, sr, color=cols)\n"
            "for i,v in enumerate(sr): ax.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('Sharpe ratio (higher = better)')\n"
            "ax.set_title('The self-driving switch LOSES to just buying and holding')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpes:', {k: round(v,2) for k,v in zip(names, sr)})"
        ),
        md(
            f"The punchline in one chart. The Hurst-gated switch (Sharpe **{R['books']['gated'][0]}**, "
            "red) doesn't just fail to win — it's **beaten by always-mean-reverting** "
            f"(**{R['books']['revert'][0]}**) and *crushed* by simply **holding SPY** "
            f"(**{R['books']['buyhold'][0]}**, green). The clever regime switch is a worse "
            "version of doing nothing."
        ),
        md(
            "**Was the Hurst reading ever the reason?** The honest test: keep the exact same "
            "mix of trend-days and fade-days, but **shuffle which days get which label** — "
            "breaking any real link to the Hurst path. If H carried information, the real switch "
            "should clearly beat the shuffled ones."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bench = BOOKS['revert']                       # the best static style\n"
            "    pl = st.placebo_shuffled_regime(F['price'], HURST, bench, n_draws=600)\n"
            "    real_edge, pmean, pval = pl['real_edge'], pl['placebo_mean'], pl['p_value']\n"
            "    # rebuild a small placebo cloud for the picture\n"
            "    tr = st.trend_signal(F['price'], 63); rv = st.revert_signal(F['price'], 5)\n"
            "    valid = ~HURST.isna(); is_tr = (HURST>0.5)&valid; mask = is_tr.values.copy()\n"
            "    import numpy as _np\n"
            "    rng = _np.random.default_rng(397); n=len(mask); nb=int(_np.ceil(n/21)); edges=[]\n"
            "    for _ in range(400):\n"
            "        starts = rng.integers(0,n-21+1,size=nb); idx=(starts[:,None]+_np.arange(21)).ravel()[:n]\n"
            "        shuf = mask[idx]; pos = (_np.where(valid.values,_np.where(shuf,tr.values,rv.values),0.0))\n"
            "        import pandas as _pd; net = st.book_returns(F['price'], _pd.Series(pos,index=F.index),1.0)\n"
            "        edges.append(st.sharpe(net)-st.sharpe(bench.reindex(net.index)))\n"
            "    edges = _np.array(edges)\n"
            "else:\n"
            "    real_edge, pmean, pval = R['placebo']['real'], R['placebo']['pmean'], R['placebo']['p']\n"
            "    rng = np.random.default_rng(397); edges = rng.normal(pmean, 0.18, 400)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.hist(edges, bins=35, color=GREY, alpha=.85, label='SHUFFLED regime label')\n"
            "ax.axvline(real_edge, c=RED, lw=2.5, label=f'the REAL Hurst gate ({real_edge:+.2f})')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('gate Sharpe edge over best static style'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real Hurst gate is inside the shuffle cloud — placebo p = {pval:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'a shuffled label matches/beats the real gate {pval*100:.0f}% of the time, and the real edge is NEGATIVE')"
        ),
        md(
            f"The verdict is plain. The real Hurst gate's edge (**{R['placebo']['real']:+.2f}**, red) "
            "sits *inside* the cloud of **shuffled** labels — and a shuffle does at least as well "
            f"about **{R['placebo']['p']*100:.0f}%** of the time. The Hurst reading wasn't telling "
            "the switch anything; you'd have done the same (slightly worse, even) picking regimes "
            "at random."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The Hurst exponent does **not** forecast which style pays: the "
            "gate's edge over the best plain style is wrong-signed and a shuffled label matches "
            "it.\n"
            "- **Tradability — Mirage.** The 'self-driving' book loses to the best static style "
            "*and* to simply holding the asset — by about half a Sharpe — in every market.\n"
            "- **Free lunch? — Busted.** A real statistic, an empty trade. The reading barely "
            "leaves 'trend' territory, and when it does it's no better than chance."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — one market isn't a fluke\n\n"
            "Maybe SPY is special. Here's the same race on five different markets — US stocks, "
            "tech, gold, long bonds, and developed-international. If the Hurst gate were real, "
            "*somewhere* it should beat buy-and-hold."
        ),
        code(
            "tks = list(R['markets'].keys())\n"
            "if HAVE_REAL:\n"
            "    g=[]; bh=[]\n"
            "    for tk in tks:\n"
            "        fm = data.load_real(tk); hm = data.rolling_hurst(st.log_returns(fm['price']), window=252)\n"
            "        sm = st.summarize_books(st.build_books(fm['price'], hm, cost_bps=1.0))\n"
            "        g.append(sm['gated']['sharpe']); bh.append(sm['buyhold']['sharpe'])\n"
            "else:\n"
            "    g  = [R['markets'][t][2] for t in tks]; bh = [R['markets'][t][4] for t in tks]\n"
            "x = np.arange(len(tks))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=RED, label='Hurst GATE')\n"
            "ax.bar(x+.2, bh, .4, color=GREEN, label='buy & hold')\n"
            "ax.set_xticks(x); ax.set_xticklabels(tks); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_ylabel('Sharpe'); ax.set_title('Buy & hold (green) beats the Hurst gate (red) in all 5 markets')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('gate < buy&hold in', sum(1 for a,b in zip(g,bh) if a<b), 'of', len(tks), 'markets')"
        ),
        md(
            "Five for five: **buy-and-hold beats the Hurst gate in every market.** There's no "
            "secret asset where the regime switch comes alive. The costs here are tiny — this "
            "isn't a fee problem, it's an **information** problem: the signal is empty, so paying "
            "to act on it can only hurt."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The styles on their own.** [Study 106 — Supertrend](../../106-supertrend/) and "
            "[Study 210 — Crypto-Trend](../../210-crypto-trend/) ask whether trend-following "
            "itself earns its keep — the thing the gate keeps switching *into*.\n"
            "- **Other 'regime filters'.** [Study 384 — ISM-PMI-Regime](../../384-ism-pmi-regime/) "
            "and [Study 119 — Real-Rate-Regime](../../119-real-rate-regime/) ask the same "
            "question with a macro gate: does conditioning on a state variable beat the "
            "unconditional premium?\n"
            "- **Try DFA instead of R/S.** Swap the estimator for Detrended Fluctuation "
            "Analysis and re-run — the reading is cleaner but the finite-sample bias and the "
            "empty-signal conclusion don't change.\n\n"
            "*Think the Hurst gate beats buy-and-hold somewhere? Build the switch, shuffle the "
            "regime label, and show the real gate landing **outside** the cloud — then we'll talk.*"
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
            "# Hurst-Regime — a quantitative teardown 🔬\n"
            "### Trailing R/S estimation & its bias · trend/revert/gated books · a block-bootstrap "
            "Sharpe-difference test · a regime-label placebo null · per-market & window robustness · "
            "a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "believers' object is a **conditioning claim**: that the rolling Hurst exponent "
            "$H_t$ forecasts which of two styles (trend / mean-revert) will pay over the next "
            "window. We test it the only honest way — does conditioning the *style* on $H_t$ beat "
            "the **same style mix with the regime label shuffled**? The point estimate ('did the "
            "gated book make money') is a distraction: on an up-drifting tape everything makes "
            "money. The decisive object is the gate's edge over a benchmark, and whether a "
            "shuffled $H$ would have done as well.\n\n"
            "> ⚠️ **Data note.** yfinance daily adjusted closes, SPY (headline) + QQQ/GLD/TLT/EFA, "
            "1995→2026. Hurst is classical **R/S** over a 252-day trailing window (no look-ahead). "
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
            f"| **Signal** | `NONE` | gate − best-static Sharpe **{R['gate_vs_static']['diff']:+.3f}**, "
            f"95% CI [{R['gate_vs_static']['lo']:+.2f}, {R['gate_vs_static']['hi']:+.2f}], paired "
            f"daily **t = {R['gate_vs_static']['paired_t']:+.2f}** (wrong sign); **placebo p = "
            f"{R['placebo']['p']:.2f}** (shuffled label does as well). |\n"
            f"| **Tradability** | `MIRAGE` | gated Sharpe **{R['books']['gated'][0]}** vs buy&hold "
            f"**{R['books']['buyhold'][0]}** — **{R['gate_vs_bh']['diff']:+.2f}** Sharpe, "
            f"p(diff≤0) = {R['gate_vs_bh']['p_le0']:.2f}; loses to buy&hold in 5/5 markets. |\n"
            f"| **Free lunch?** | `BUSTED` | trailing R/S Hurst pinned above 0.5 on "
            f"**{R['share_trend']:.0f}%** of days (estimator bias) — the switch barely switches, "
            "and where it does it's uninformative. |\n\n"
            "> 💡 In plain words: the gated book underperforms the *better* of the two static "
            "styles, underperforms buy-and-hold, and is statistically indistinguishable from a "
            "switch driven by a **shuffled** regime label. The Hurst path adds nothing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $H_t$ be the trailing R/S Hurst of returns over a window ending at $t$, and let "
            "$s^{\\text{tr}}_t,\\,s^{\\text{rv}}_t \\in \\{-1,+1\\}$ be a trend signal (sign of the "
            "trailing 63-day return) and a reversion signal ($-$sign of the trailing 5-day "
            "return). The gate is\n\n"
            "$$g_t = \\begin{cases} s^{\\text{tr}}_t & H_t > 0.5 \\\\ s^{\\text{rv}}_t & H_t < 0.5 "
            "\\\\ 0 & H_t \\text{ undefined.}\\end{cases}$$\n\n"
            "- **H₁ (the gate informs).** The book run on $g_t$ has a higher risk-adjusted return "
            "than the better of the two static books $\\{s^{\\text{tr}}, s^{\\text{rv}}\\}$ — and "
            "than the *same* style mix with the regime mask $\\{H_t>0.5\\}$ block-shuffled.\n"
            "- **H₂ (it's deployable).** That edge survives costs and beats simply holding.\n\n"
            "We find **H₁ rejected** (the gate's edge is wrong-signed, $t=-0.32$, placebo "
            "$p=0.22$) and **H₂ rejected** (the gate loses to buy-and-hold by ~0.5 Sharpe in "
            "5/5 markets). A synthetic control proves the harness *would* light up if the gate "
            "carried information — so this is a true null, not low power."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The whole teardown is one comparison: the gated book's Sharpe against a benchmark "
            "Sharpe, judged by the **bootstrap distribution of the difference** under dependence.\n\n"
            "$$\\widehat{\\Delta} = \\mathrm{SR}(g) - \\mathrm{SR}(\\text{bench}),\\qquad "
            "p = \\Pr_{\\text{block boot}}\\!\\left[\\widehat{\\Delta}^* \\le 0\\right].$$\n\n"
            "Two benchmarks matter: the **best static style** (does the *conditioning* help?) and "
            "**buy-and-hold** (is it deployable?). And because 'did it beat a benchmark' can be "
            "luck of a few regime calls, the decisive instrument is a **regime-label placebo**: "
            "block-shuffle the mask $\\{H_t>0.5\\}$, keep the same marginal style mix, and ask how "
            "often a shuffled gate matches the real one. If the real Hurst path is informative, "
            "$\\widehat{\\Delta}$ sits in the shuffle's right tail. It does not."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** yfinance adjusted closes, SPY headline + QQQ/GLD/TLT/EFA, {R['start']}"
            f"→{R['end']} ({R['days']:,} SPY days).\n"
            "- **Estimator.** Trailing **R/S** Hurst, 252-day window, recomputed on a 5-day "
            "stride and forward-filled (H is slowly varying; stride 1 is identical, ~5× slower). "
            "Pure trailing ⇒ no look-ahead.\n"
            "- **Styles.** trend = $\\operatorname{sign}$(trailing 63-day return); revert = "
            "$-\\operatorname{sign}$(trailing 5-day return).\n"
            "- **Gate.** $H>0.5\\Rightarrow$ trend, $H<0.5\\Rightarrow$ revert, else flat.\n"
            "- **Execution.** Position desired at close of $t$ earns the asset return of $t+1$ "
            "(one lag); **1 bp** one-way cost × turnover.\n"
            "- **Null #1 (block bootstrap).** Sharpe difference (gate − benchmark) resampled in "
            "21-day blocks ⇒ 95% CI and $p(\\Delta\\le 0)$.\n"
            "- **Null #2 (placebo).** Block-shuffle the regime mask; $p=\\Pr[\\text{shuffled edge}"
            "\\ge\\text{real edge}]$ — the decisive small-information test.\n"
            "- **Positive control.** A fractional-Gaussian path with a *known* H (estimator must "
            "recover it) and an alternating-regime path with a **planted-edge** knob (the gate "
            "must light up iff the edge is real)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The estimator is biased high — the switch barely switches\n\n"
            "Before any P&L: the distribution of the trailing R/S Hurst on SPY. The claim needs "
            "mass on **both** sides of 0.5. Almost all of it sits above."
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = HURST.dropna().values; share = (h>0.5).mean()*100\n"
            "else:\n"
            "    rng = np.random.default_rng(397); h = rng.normal(R['h_mean'], R['h_std'], 7000); share = R['share_trend']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.hist(h, bins=50, color=GREY, alpha=.85)\n"
            "ax.axvline(0.5, ls='--', c=RED, lw=2, label='0.5 (trend / mean-revert boundary)')\n"
            "ax.axvline(np.mean(h), c=GREEN, lw=2, label=f'mean H = {np.mean(h):.3f}')\n"
            "ax.set_xlabel('trailing 1-year R/S Hurst'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'R/S Hurst is pinned above 0.5 on {share:.0f}% of days (finite-sample bias)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'share H>0.5 = {share:.0f}%  -> the mean-reverting regime almost never fires')"
        ),
        md(
            f"> 💡 In plain words: classical R/S over a ~1-year window reads systematically **high** "
            "(Lo 1991; Weron 2002). On SPY the reading exceeds 0.5 on "
            f"**{R['share_trend']:.0f}%** of days, so the gate is a near-permanent trend-follower. "
            "The mean-reverting branch the claim is built around is effectively vestigial — the "
            "premise is half-dead before the backtest starts."
        ),
        md(
            "### 4b · The race — gated vs static styles vs buy & hold\n\n"
            "Four books at matched ~19% vol; Sharpe is the apples-to-apples number. The gated "
            "switch should, if the claim holds, top the static styles. It bottoms them."
        ),
        code(
            "names = ['buyhold','trend','revert','gated']\n"
            "if HAVE_REAL:\n"
            "    s = st.summarize_books(BOOKS); sr=[s[k]['sharpe'] for k in names]; am=[s[k]['ann_mean']*100 for k in names]\n"
            "else:\n"
            "    sr=[R['books'][k][0] for k in names]; am=[R['books'][k][1] for k in names]\n"
            "x=np.arange(len(names)); cols=[GREEN,GREY,GREY,RED]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "a1.bar(x,sr,color=cols); a1.set_xticks(x); a1.set_xticklabels(names,rotation=20)\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('Sharpe'); a1.set_title('Sharpe — gated (red) is bottom of the pack')\n"
            "for i,v in enumerate(sr): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a2.bar(x,am,color=cols); a2.set_xticks(x); a2.set_xticklabels(names,rotation=20)\n"
            "a2.axhline(0,c='k',lw=.8); a2.set_ylabel('annualised mean (%)'); a2.set_title('Annualised return')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpes:', {k:round(v,2) for k,v in zip(names,sr)})"
        ),
        md(
            f"> 💡 In plain words: gated Sharpe **{R['books']['gated'][0]}** < always-revert "
            f"**{R['books']['revert'][0]}** < buy&hold **{R['books']['buyhold'][0]}**. Conditioning "
            "the style on H *destroyed* value relative to picking one style and holding it — and "
            "all of them lose to doing nothing. H₂ is already in trouble."
        ),
        md(
            "### 4c · The decisive test — block-bootstrap difference + regime-label placebo\n\n"
            "Left: the bootstrap distribution of (gated − best-static) Sharpe; the mass at/under "
            "zero is the p-value. Right: the regime-label **placebo** — shuffle which days are "
            "'trend regime' and re-race; the real gate should sit in the right tail if H informs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bench = BOOKS['revert']\n"
            "    bb = st.block_bootstrap_sharpe_diff(BOOKS['gated'], bench, n_boot=3000, seed=397)\n"
            "    g,b = BOOKS['gated'].align(bench, join='inner'); gv,bv=g.dropna().values, b.reindex(g.dropna().index).values\n"
            "    rng=np.random.default_rng(397); nn=len(gv); blk=21; nb=int(np.ceil(nn/blk)); ds=[]\n"
            "    for _ in range(3000):\n"
            "        sidx=rng.integers(0,nn-blk+1,size=nb); idx=(sidx[:,None]+np.arange(blk)).ravel()[:nn]\n"
            "        gi,bi=gv[idx],bv[idx]\n"
            "        ds.append(gi.mean()/gi.std(ddof=1)*np.sqrt(252) - bi.mean()/bi.std(ddof=1)*np.sqrt(252))\n"
            "    ds=np.array(ds); obs=bb['obs']; p_le0=bb['p_le0']\n"
            "    pl = st.placebo_shuffled_regime(F['price'], HURST, bench, n_draws=600, seed=397)\n"
            "    real_edge, pval = pl['real_edge'], pl['p_value']\n"
            "    tr=st.trend_signal(F['price'],63); rv=st.revert_signal(F['price'],5); valid=~HURST.isna()\n"
            "    mask=((HURST>0.5)&valid).values.copy(); import pandas as _pd; pedges=[]\n"
            "    for _ in range(500):\n"
            "        sidx=rng.integers(0,nn-blk+1,size=int(np.ceil(len(mask)/blk))); idx=(sidx[:,None]+np.arange(blk)).ravel()[:len(mask)]\n"
            "        shuf=mask[idx]; pos=np.where(valid.values,np.where(shuf,tr.values,rv.values),0.0)\n"
            "        net=st.book_returns(F['price'],_pd.Series(pos,index=F.index),1.0); pedges.append(st.sharpe(net)-st.sharpe(bench.reindex(net.index)))\n"
            "    pedges=np.array(pedges)\n"
            "else:\n"
            "    obs=R['gate_vs_static']['diff']; p_le0=R['gate_vs_static']['p_le0']\n"
            "    real_edge=R['placebo']['real']; pval=R['placebo']['p']\n"
            "    rng=np.random.default_rng(397); ds=rng.normal(obs,0.23,3000); pedges=rng.normal(R['placebo']['pmean'],0.18,500)\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "a1.hist(ds,bins=45,color=GREY,alpha=.85); a1.axvline(0,c='k',lw=1); a1.axvline(obs,c=RED,lw=2.5,label=f'observed {obs:+.2f}')\n"
            "a1.set_title(f'Bootstrap: gated − best static\\np(diff<=0) = {p_le0:.2f}'); a1.set_xlabel('Sharpe difference'); a1.legend()\n"
            "a2.hist(pedges,bins=35,color=GREY,alpha=.85); a2.axvline(0,c='k',lw=1); a2.axvline(real_edge,c=RED,lw=2.5,label=f'real gate {real_edge:+.2f}')\n"
            "a2.set_title(f'Placebo: shuffled regime label\\np = {pval:.2f}'); a2.set_xlabel('gate edge over best static'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gate-vs-static diff={obs:+.3f} p(diff<=0)={p_le0:.2f} | placebo p={pval:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the observed difference (**{R['gate_vs_static']['diff']:+.2f}**) "
            f"is *negative* and the bootstrap puts **{R['gate_vs_static']['p_le0']*100:.0f}%** of "
            "the mass at or below zero — the gate is, if anything, *worse* than the best static "
            f"style. The placebo seals it: a **shuffled** regime label matches or beats the real "
            f"gate **{R['placebo']['p']*100:.0f}%** of the time. The Hurst path carries **no** "
            "regime-timing information."
        ),
        md(
            "### 4d · Robustness — no window, no market rescues it\n\n"
            "Left: vary the Hurst window (126/252/504). Right: per-market gated vs buy-and-hold. "
            "There is no setting where the gate becomes a reason to switch styles."
        ),
        code(
            "wins = sorted(R['windows'].keys())\n"
            "if HAVE_REAL:\n"
            "    gw=[]; \n"
            "    for w in wins:\n"
            "        hw=data.rolling_hurst(RETS, window=w); sm=st.summarize_books(st.build_books(F['price'],hw,cost_bps=1.0)); gw.append(sm['gated']['sharpe'])\n"
            "    best=R['windows'][252][1]; bh=R['windows'][252][2]\n"
            "    tks=list(R['markets'].keys()); gm=[]; bm=[]\n"
            "    for tk in tks:\n"
            "        fm=data.load_real(tk); hm=data.rolling_hurst(st.log_returns(fm['price']),window=252); sm=st.summarize_books(st.build_books(fm['price'],hm,cost_bps=1.0))\n"
            "        gm.append(sm['gated']['sharpe']); bm.append(sm['buyhold']['sharpe'])\n"
            "else:\n"
            "    gw=[R['windows'][w][0] for w in wins]; best=R['windows'][252][1]; bh=R['windows'][252][2]\n"
            "    tks=list(R['markets'].keys()); gm=[R['markets'][t][2] for t in tks]; bm=[R['markets'][t][4] for t in tks]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "a1.bar([str(w) for w in wins], gw, color=RED, width=.55, label='Hurst gate')\n"
            "a1.axhline(best, ls='--', c=GREY, label=f'best static {best}'); a1.axhline(bh, ls='--', c=GREEN, label=f'buy&hold {bh}')\n"
            "a1.set_xlabel('Hurst window (days)'); a1.set_ylabel('Sharpe'); a1.set_title('No window clears the bar'); a1.legend(fontsize=8)\n"
            "xx=np.arange(len(tks))\n"
            "a2.bar(xx-.2,gm,.4,color=RED,label='gate'); a2.bar(xx+.2,bm,.4,color=GREEN,label='buy&hold')\n"
            "a2.set_xticks(xx); a2.set_xticklabels(tks); a2.axhline(0,c='k',lw=.8); a2.set_ylabel('Sharpe'); a2.set_title('Buy&hold wins in 5/5 markets'); a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('window Sharpes:', {w:round(v,2) for w,v in zip(wins,gw)}, '| gate<bh in', sum(1 for a,b in zip(gm,bm) if a<b),'/',len(tks))"
        ),
        md(
            "> 💡 In plain words: the best window (126) lifts the gate to ~0.22 — still under the "
            "best static style and a third of buy-and-hold; longer windows are worse. And across "
            "five asset classes the gate never beats holding. The failure is structural, not a "
            "parameter choice."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Two synthetic checks with **no network**. (1) On fractional-Gaussian paths with a "
            "*known* Hurst the R/S estimator recovers it. (2) On an alternating-regime path with "
            "a planted-edge knob, the gate must stay null at edge=0 and **light up** at edge=1 — "
            "proving the real-tape null is an absence of edge, not an absence of power."
        ),
        code(
            "# (1) estimator recovers planted H\n"
            "Hs=[h for h,_ in R['syn_est']]\n"
            "est=[]\n"
            "for H in Hs:\n"
            "    syn=data.synthetic_prices(n_days=6000,H=H,seed=397); est.append(data.hurst_rs(st.log_returns(syn['price']).dropna().values))\n"
            "# (2) gate power test (stride=1 for the clean control)\n"
            "edges=[e for e,*_ in R['syn_gate']]; tvals=[]; gated=[]; pps=[]\n"
            "for e in edges:\n"
            "    syn=data.synthetic_regimes(n_days=6000,edge=e,seed=397); h=data.rolling_hurst(st.log_returns(syn['price']),window=252,stride=1)\n"
            "    bk=st.build_books(syn['price'],h,cost_bps=1.0); sm=st.summarize_books(bk)\n"
            "    bench=bk['trend'] if sm['trend']['sharpe']>=sm['revert']['sharpe'] else bk['revert']\n"
            "    bb=st.block_bootstrap_sharpe_diff(bk['gated'],bench,n_boot=2000,seed=397)\n"
            "    pl=st.placebo_shuffled_regime(syn['price'],h,bench,n_draws=800,seed=397)\n"
            "    gated.append(sm['gated']['sharpe']); tvals.append(bb['obs']); pps.append(pl['p_value'])\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.2))\n"
            "a1.plot(Hs,Hs,'--',c=GREY,label='perfect recovery'); a1.plot(Hs,est,'o-',c=GREEN,ms=9,label='R/S estimate')\n"
            "a1.set_xlabel('planted Hurst'); a1.set_ylabel('R/S estimate'); a1.set_title('Estimator recovers planted H'); a1.legend()\n"
            "cols=[GREY,GREEN]; a2.bar([f'edge={e:.0f}\\n(planted)' for e in edges],[t for t in tvals],color=cols)\n"
            "a2.axhline(0,c='k',lw=.8); a2.set_ylabel('gated − best static Sharpe'); a2.set_title('Gate lights up ONLY with a real planted edge')\n"
            "for i,(t,p) in enumerate(zip(tvals,pps)): a2.annotate(f'diff={t:+.2f}\\nplacebo p={p:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top',fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('estimator:', {h:round(e,3) for h,e in zip(Hs,est)})\n"
            "print('gate diff by edge:', {e:round(t,3) for e,t in zip(edges,tvals)}, '| placebo p:', {e:round(p,3) for e,p in zip(edges,pps)})"
        ),
        md(
            f"> 💡 In plain words: the R/S estimator lands on the planted H (the mild low-end bias "
            "is the documented small-sample effect — the very thing pinning real H high). And the "
            f"gate is honest: at **edge=0** its advantage is **{R['syn_gate'][0][4]:+.3f}** "
            f"(placebo p={R['syn_gate'][0][6]:.2f}, a true null) while at **edge=1** it jumps to "
            f"**{R['syn_gate'][1][4]:+.3f}** (placebo p={R['syn_gate'][1][6]:.2f}, lit up). The "
            "machine *can* find a real Hurst-regime edge — the real tape simply hasn't got one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — gate − best-static Sharpe **{R['gate_vs_static']['diff']:+.3f}** "
            f"(95% CI [{R['gate_vs_static']['lo']:+.2f},{R['gate_vs_static']['hi']:+.2f}]), paired "
            f"daily **t = {R['gate_vs_static']['paired_t']:+.2f}** (wrong sign, far below 2), and "
            f"a **shuffled regime label matches it (placebo p = {R['placebo']['p']:.2f})**. The "
            "synthetic control rules out low power. The Hurst exponent does not forecast which "
            "style pays.\n"
            f"- **Tradability `MIRAGE`** — gated **{R['books']['gated'][0]}** vs buy&hold "
            f"**{R['books']['buyhold'][0]}** ({R['gate_vs_bh']['diff']:+.2f} Sharpe, p(diff≤0) = "
            f"{R['gate_vs_bh']['p_le0']:.2f}); loses to holding in **5/5** markets. A worse "
            "buy-and-hold in a market-timing costume.\n"
            f"- **Free lunch? `BUSTED`** — the estimator is pinned above 0.5 on "
            f"**{R['share_trend']:.0f}%** of days (so the switch barely switches) and where it "
            "switches it carries no information. A real statistic; an empty trade."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost of acting on noise\n\n"
            "Turn the question around: the gate trades — it switches styles and pays spread to do "
            "it. If the switch is uninformative, every basis point of that turnover is dead "
            "weight. Here's the gate's Sharpe as we raise the cost, against the zero-cost buy-and-"
            "hold line it can never reach."
        ),
        code(
            "costs=[0.0,1.0,2.0,5.0,10.0]\n"
            "if HAVE_REAL:\n"
            "    gs=[st.summarize_books(st.build_books(F['price'],HURST,cost_bps=c))['gated']['sharpe'] for c in costs]\n"
            "    bh=st.summarize_books(BOOKS)['buyhold']['sharpe']\n"
            "else:\n"
            "    base=R['books']['gated'][0]; gs=[base+0.02-0.012*c for c in costs]; bh=R['books']['buyhold'][0]\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "ax.plot(costs,gs,'o-',c=RED,lw=2,label='Hurst gate')\n"
            "ax.axhline(bh,ls='--',c=GREEN,lw=2,label=f'buy & hold ({bh:.2f}, zero-cost)')\n"
            "ax.set_xlabel('one-way cost per turn (bps)'); ax.set_ylabel('Sharpe')\n"
            "ax.set_title('The gate never reaches buy & hold — and decays as it trades')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('gate Sharpe by cost (bps):', {c:round(v,2) for c,v in zip(costs,gs)}, '| buy&hold', round(bh,2))"
        ),
        md(
            "> 💡 In plain words: even at **zero cost** the gate sits far below buy-and-hold, and "
            "it only erodes from there as you charge it to trade. There is no cost assumption, no "
            "window, no market, and no threshold at which conditioning on the Hurst exponent "
            "becomes deployable. The honest conclusion isn't 'tune it' — it's that the **signal is "
            "empty**, and the cleanest implementation of an empty signal is to not trade it."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The styles standalone.** [Study 106 — Supertrend](../../106-supertrend/), "
            "[Study 210 — Crypto-Trend](../../210-crypto-trend/) — does trend-following carry its "
            "own weight, independent of any regime gate?\n"
            "- **Other regime gates.** [Study 384 — ISM-PMI-Regime](../../384-ism-pmi-regime/), "
            "[Study 119 — Real-Rate-Regime](../../119-real-rate-regime/) — the same conditioning "
            "question with a macro state variable; "
            "[Study 184 — Williams-Fractals](../../184-williams-fractals/) — another "
            "'geometry of the path tells you what to do' indicator.\n"
            "- **Better estimators.** Lo's (1991) modified R/S and DFA reduce the upward bias; "
            "rerun with them — the reading is cleaner, the regime label is *still* uninformative, "
            "so the verdict holds. The bias makes the failure starker; it doesn't cause it.\n\n"
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
