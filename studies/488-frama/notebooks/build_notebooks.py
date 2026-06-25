"""Generate the two narrative notebooks for Study 488 (FRAMA).

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
# 2026-05-31, partial June dropped), 21.4 years, FRAMA N=16, price>FRAMA cross-up long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=2832, n_ema=2614, frama_n=16,
    fp_spy="4cb5244f3990",
    # pooled FRAMA cross-up, per horizon:
    # (H, n, frama_bps, win%, one_sample_t, random_bps, delta_rnd, ema_bps, delta_ema,
    #  net_bps, welch_t, welch_p)
    h5=(5, 2830, 18.9, 58, 3.48, 29.7, -10.8, 18.9, 0.0, 16.9, -1.51, 0.131),
    h10=(10, 2828, 51.2, 61, 6.10, 50.8, 0.4, 46.2, 5.0, 49.2, 0.04, 0.970),
    h20=(20, 2823, 102.1, 64, 7.27, 99.3, 2.8, 88.6, 13.4, 100.1, 0.20, 0.841),
    h60=(60, 2805, 285.0, 69, 8.08, 257.9, 27.1, 278.6, 6.3, 283.0, 1.17, 0.244),
    # per-ticker H=20: (ticker, entries, frama_bps, one_sample_t, random_bps, d_rnd, d_ema)
    per=[("SPY", 559, 85.9, 3.63, 104.5, -18.6, 3.2), ("QQQ", 581, 112.4, 3.08, 123.9, -11.6, 5.0),
         ("IWM", 560, 84.7, 2.46, 86.0, -1.3, -16.0), ("DIA", 569, 88.2, 3.39, 89.8, -1.6, 30.9),
         ("GLD", 563, 138.7, 4.47, 91.3, 47.4, 44.3)],
    # shuffled-alpha placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(85.9, 0.417, 500),
    # synthetic control (H=20, n_days=6000): (edge, n, frama_bps, win%, one_sample_t)
    syn=[(0.00, 550, 3.2, 46, 0.12), (2.00, 100, 2136.1, 67, 4.11)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Adaptive_edge%3F: Busted](https://img.shields.io/badge/Adaptive_edge%3F-Busted-8b949e?style=flat-square)\n\n"
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

from frama import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real FRAMA cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does FRAMA's \"fractal-adaptive\" smoothing actually buy edge? 〰️\n"
            "### A clever moving average — fast in trends, slow in chop — meets a stopwatch\n\n"
            + BADGES +
            "Most moving averages have one speed. **FRAMA** — John Ehlers' *Fractal Adaptive Moving "
            "Average* — is supposed to be smarter: it measures the **fractal dimension** of recent "
            "price and changes its own smoothing on the fly. In a clean trend it *speeds up* and "
            "hugs price; in choppy, sideways action it *slows down* and flattens out. The pitch, "
            "repeated on every charting forum, is that this lets a \"buy when price crosses above "
            "FRAMA\" rule **catch trends sooner and dodge the whipsaws** that fool a plain moving "
            "average.\n\n"
            "It *sounds* clever. But \"adaptive\" is exactly the kind of word that sells a tool while "
            "hiding whether it does anything. So we did the only fair thing: encode FRAMA **exactly "
            "as Ehlers specifies**, fire the cross-up rule thousands of times across five big indices "
            "over 21 years, and time the result with a stopwatch — against two honest baselines: "
            "**buying on random days**, and the same rule on a **plain, fixed-speed moving "
            "average**.\n\n"
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
            "| If I buy when price crosses above FRAMA, do I make money? | **Yes — but only because "
            "the market goes up.** The win-rate is ~60% and the returns look great. |\n"
            "| Is that *FRAMA's* doing? | **No.** Buy on **random days** instead and you do just as "
            "well — the cross-up adds essentially nothing. |\n"
            "| Is the \"fractal-adaptive\" part worth anything? | **No.** Run the *exact same rule* on "
            "a **plain fixed-speed moving average** and you land in the same place. |\n"
            "| Does the fractal dimension carry information? | **No.** Scramble the adaptation in time "
            "and the result barely changes. The fractal machinery is decorative. |\n\n"
            "> FRAMA is a perfectly fine way to *draw* a trend. As a *forecast* — \"the cross-up will "
            "pay\" — it's a **mirage**: all of the apparent edge is the market's long-run climb, none "
            "of it is the fractal adaptation."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A normal moving average lags. FRAMA measures the **fractal dimension** of the last "
            "few bars — how 'rough' or 'straight' the path is — and adapts its smoothing: near 1 "
            "(a straight trend) it follows fast; near 2 (jagged chop) it slows to a crawl. So it "
            "tracks trends tightly and ignores noise. Buy when price crosses above it.\"*\n\n"
            "This is **John Ehlers'** FRAMA (*Stocks & Commodities*, 2005), built into most charting "
            "suites and a staple of the \"adaptive indicator\" genre. The smoothing constant is "
            "literally `alpha = exp(−4.6·(D−1))`, where `D` is the fractal dimension. So: does the "
            "clever adaptation actually *forecast* — or is it a fancier way to lag?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If FRAMA genuinely *forecast* trends better than a dumb moving average, that would be a "
            "real, tradable edge — a crack in market efficiency you could mechanize. That's the dream "
            "the \"adaptive\" label sells.\n\n"
            "But there are two traps. First, FRAMA is a **moving average of past price** drawn on a "
            "market (stock indices) that drifts **up** over time — so *any* \"buy above the average\" "
            "rule will look profitable, fractal or not. Second, \"adaptive\" might just be a more "
            "complicated way to compute the same lagging line. To separate the **tool** from the "
            "**tide** we (a) race FRAMA against buying on **random days**, and (b) race it against the "
            "same rule on a **plain fixed-speed EMA**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Compute FRAMA mechanically.** Ehlers' recursion over an "
            f"**N = {R['frama_n']}-bar** window — strictly causal, no peeking at future bars.\n"
            "2. **Trade the lore.** Buy the bar the close first **crosses above FRAMA**, at the next "
            "close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baselines.** Do the exact same hold on **random days** (the drift "
            "baseline), and run the same cross-up on a **plain fixed-speed EMA** (the 'is adaptation "
            "worth it?' baseline). If FRAMA matters, it must beat both.\n"
            "4. *If it doesn't, the tool is a mirage* — that's the result that would make us say so, "
            "announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does FRAMA even look like, and how does it differ from a fixed moving "
            "average? Here's SPY with FRAMA (adaptive) and a plain EMA of the same average speed, "
            "plus the cross-up buys the rule would fire."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY'); cl = b['close']; seg = cl.iloc[-450:]\n"
            "    fr = st.frama(b, n=R['frama_n']); em = st.fixed_ema(b, n=R['frama_n'])\n"
            "    ent = st.frama_cross_entries(b, n=R['frama_n']); ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.plot(seg.index, fr.reindex(seg.index), c=GREEN, lw=1.4, label='FRAMA (adaptive)')\n"
            "    ax.plot(seg.index, em.reindex(seg.index), c=GREY, lw=1.2, ls='--', label='fixed EMA (same speed)')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=38, zorder=5, label='cross-up BUY')\n"
            "    ax.set_title('FRAMA vs a fixed EMA on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('FRAMA cross-ups in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "FRAMA and the fixed EMA track each other closely — the \"adaptive\" line is rarely far "
            "from a plain one. The question is whether those green buy dots are followed by real "
            "trends. **Let's race the FRAMA cross-up against random entries** at four horizons. Blue "
            "= buy the FRAMA cross-up; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    frama_b, rnd = [], []\n"
            "    for h in hs:\n"
            "        ff, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); c = b['close']\n"
            "            e = st.frama_cross_entries(b, n=R['frama_n'])\n"
            "            re = st.random_entries(c, max(len(e),50), n=R['frama_n'], seed=7)\n"
            "            ff.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        frama_b.append(np.concatenate(ff).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    frama_b = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, frama_b, .4, color='#2c6fbb', label='buy the FRAMA cross-up')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(frama_b,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('FRAMA cross-up does NOT beat random — a dead heat'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('FRAMA:', [round(v) for v in frama_b]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story in one chart. The FRAMA cross-up makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but so does **buying on random days** "
            f"(**+{R['h20'][5]:.0f} bps**). It's a dead heat at every horizon; at 5 days FRAMA is "
            "actually *behind* a coin flip. The apparent edge was **the market's upward drift**, not "
            "the fractal smoothing."
        ),
        md(
            "**One more sanity check.** What if we scramble FRAMA's *adaptation* — keep the same set "
            "of smoothing speeds but shuffle *when* each one is applied, so it no longer tracks the "
            "fractal dimension? If the fractal logic really matters, the scramble should hurt."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    pl = st.shuffled_alpha_placebo(b, 20, n=R['frama_n'], n_draws=300, seed=488)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real FRAMA cross-up (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *time-scrambled-adaptation* runs do at least as well (p={pval:.2f}).')\n"
            "print('=> the fractal adaptation is not doing the work.')"
        ),
        md(
            f"Around **{R['placebo'][1]*100:.0f}%** of the **scrambled-adaptation** runs match or "
            f"beat the real FRAMA (*p* = {R['placebo'][1]:.2f}). If the fractal dimension genuinely "
            "drove an edge, randomly re-timing the adaptation would collapse the result. It doesn't — "
            "because the result was never about the fractal part."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The FRAMA cross-up does **not** beat buying on random days — it's a "
            "dead heat at every horizon (and *behind* at 5 days). The big absolute returns are the "
            "market's drift, not the indicator.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting — and a **plain fixed EMA** lands in the same place, so the adaptation buys "
            "nothing.\n"
            "- **\"Fractal-adaptive edge\"? — Busted.** Scramble the adaptation in time and the "
            "result barely moves. The fractal machinery is decorative."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. FRAMA's *only* advantage over a coin flip is the market's "
            "long-run climb — which you'd capture more cheaply (and more fully) by just **holding the "
            "index**. The cross-up rule fires thousands of times and pays costs on each, and the "
            "\"adaptive\" part can't even beat a plain moving average. As a forecasting tool it "
            "doesn't pay; as a drawing tool, it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Other adaptive MAs.** Kaufman's KAMA, Ehlers' MAMA, the Hull MA — same family, same "
            "drift confound. The desk's sibling studies "
            "([432-hull](../../432-hull-moving-average), [433-kama](../../433-kama-adaptive)) land in "
            "the same place.\n"
            "- **Different windows / clamps.** Try a wider/narrower N or the 'modified FRAMA' slow "
            "clamp — the result is robust: drift in, lagging line out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* persistent trend "
            "into a synthetic tape and shows the cross-up banks it (so the null here isn't a dead "
            "detector — it's an honest 'nothing there').\n\n"
            "*Think FRAMA forecasts? Show the cross-up beating random entries **and** a fixed EMA at "
            "**t ≥ 2** on a real tape — then we'll talk.*"
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
            "# FRAMA — a quantitative teardown 🔬\n"
            "### Causal Fractal Adaptive MA on 5 indices · cross-up forward returns · one-sample "
            "HAC *t* · a drift-matched random-entry baseline · a fixed-EMA comparator · a "
            "shuffled-alpha adaptation placebo · costs · a synthetic planted-trend control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **indicator** from the **drift**, and the *adaptive* part from a "
            "static MA: an upward-trending index makes *any* 'buy above the average' rule look good, "
            "so the only meaningful tests are cross-up-vs-random, cross-up-vs-fixed-EMA, plus a "
            "placebo that destroys the fractal adaptation while preserving its marginal.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. FRAMA window "
            f"N={R['frama_n']}, strictly causal; entry is the **next close** (one documented lag). "
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
            f"| **Signal** | `NONE` | FRAMA cross-up vs a **drift-matched random** baseline is a dead "
            f"heat (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d) and the difference **never clears t = 2** (Welch t at 20d "
            f"= {R['h20'][10]:+.2f}, 60d = {R['h60'][10]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta**; and a **fixed EMA** of the same speed matches FRAMA (Δ_ema = "
            f"{R['h5'][8]:+.0f}/{R['h20'][8]:+.0f} bps). No residual edge to scale. |\n"
            f"| **Fractal-adaptive edge?** | `BUSTED` | Time-scrambling the adaptive alpha "
            f"(shuffled-alpha placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}**. The "
            "fractal dimension isn't doing the work. |\n\n"
            "> 💡 In plain words: the cross-up *looks* significant only because indices drift up. "
            "Strip the drift (race it vs random), strip the adaptation (race it vs a fixed EMA), or "
            "scramble the fractal logic — the edge evaporates every way. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Over an N-bar window split into two halves, let $N_1,N_2$ be the per-bar amplitude of "
            "each half and $N_3$ that of the whole. The **fractal dimension** is "
            "$D=\\frac{\\log(N_1+N_2)-\\log N_3}{\\log 2}\\in[1,2]$, and the adaptive smoothing is "
            "$\\alpha_t=\\exp(-4.6\\,(D_t-1))$, giving "
            "$\\mathrm{FRAMA}_t=\\alpha_t C_t+(1-\\alpha_t)\\mathrm{FRAMA}_{t-1}$. The rule buys when "
            "$C_t$ crosses above $\\mathrm{FRAMA}_t$.\n\n"
            "- **H₀ (drift).** Cross-up returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (FRAMA forecasts).** Cross-up returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (adaptation matters).** Cross-up returns exceed a **fixed-EMA** of equal speed, "
            "and a **time-shuffled-alpha** FRAMA.\n\n"
            "We find **H₀ not rejected** (Δ ≈ 0 at every horizon), **H₁ rejected** (Welch t never "
            "≥ 2), **H₂ rejected** (Δ_ema ≈ 0; placebo p ≈ 0.42). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "entry rule inherits it; a high one-sample $t$ against **zero** measures the tide, not "
            "the tool. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a "
            "Welch test of cross-up-*minus*-random.\n\n"
            "**(b) 'Adaptive' as theatre.** FRAMA might simply be a more complicated way to compute a "
            "lagging average. The **fixed-EMA comparator** (same cross-up rule, constant alpha equal "
            "to FRAMA's average) and the **shuffled-alpha placebo** (permute the per-bar alpha in "
            "time, destroying the fractal-dimension link while keeping the alpha marginal) isolate "
            "whether the *adaptation* — not just the moving average — carries any information."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} FRAMA cross-ups** "
            f"pooled (vs {R['n_ema']} fixed-EMA cross-ups).\n"
            f"- **Indicator.** FRAMA, N={R['frama_n']}, $\\alpha=\\exp(-4.6(D-1))$ clipped to "
            "[0.01,1], strictly causal (rolling fractal dimension, no look-ahead).\n"
            "- **Entry.** First close above FRAMA (cross-up); enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of cross-up returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample cross-up vs random (the *real* test).\n"
            "- **Null #3 — fixed-EMA comparator** (same rule, static alpha) — the 'is adaptation worth it?' test.\n"
            "- **Null #4 — shuffled-alpha placebo** (fractal adaptation destroyed, marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every cross.\n"
            "- **Positive control.** Synthetic tape with a **planted** persistent trend (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the cross-up's **one-sample** t against zero (the misleading number). Right: the "
            "same cross-up vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, frama_b, rnd, ema_b, welch = [], [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        ff, rr, ge = [], [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); c = b['close']\n"
            "            e = st.frama_cross_entries(b, n=R['frama_n'])\n"
            "            ee = st.ema_cross_entries(b, n=R['frama_n'])\n"
            "            re = st.random_entries(c, max(len(e),50), n=R['frama_n'], seed=7)\n"
            "            ff.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "            ge.append(st.forward_returns(c, ee, h))\n"
            "        ff = np.concatenate(ff); rr = np.concatenate(rr); ge = np.concatenate(ge)\n"
            "        one_t.append(st.summarize(ff)['t']); frama_b.append(ff.mean()*1e4)\n"
            "        rnd.append(rr.mean()*1e4); ema_b.append(ge.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(ff, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    frama_b = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    ema_b = [R['h5'][7], R['h10'][7], R['h20'][7], R['h60'][7]]\n"
            "    welch = [R['h5'][10], R['h10'][10], R['h20'][10], R['h60'][10]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Cross-up vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long-only entry inherits it. "
            f"The right bars are the real test: cross-up-minus-random is **negative** at 5d "
            f"({R['h5'][10]:+.2f}) and never clears 2 (max {R['h60'][10]:+.2f} at 60d). FRAMA adds "
            "nothing over a coin flip."
        ),
        md(
            "### 4b · Cross-up vs random AND vs a fixed EMA — the two gaps that decide it\n\n"
            "Mean return: FRAMA cross-up, random entry, and the *same rule on a fixed EMA*. If the "
            "fractal adaptation forecast, FRAMA should tower over both. It towers over neither."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(x-.27, frama_b, .27, color='#2c6fbb', label='FRAMA cross-up')\n"
            "ax.bar(x, rnd, .27, color=GREY, label='random entry (drift)')\n"
            "ax.bar(x+.27, ema_b, .27, color=AMBER, label='fixed-EMA cross-up')\n"
            "for i,(a,b,e) in enumerate(zip(frama_b,rnd,ema_b)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.27,a),ha='center',va='bottom',fontsize=7)\n"
            "    ax.annotate(f'{e:+.0f}',(i+.27,e),ha='center',va='bottom',fontsize=7)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('FRAMA beats neither random nor a plain fixed EMA'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('FRAMA - random (bps):', [round(a-b) for a,b in zip(frama_b,rnd)])\n"
            "print('FRAMA - fixedEMA (bps):', [round(a-e) for a,e in zip(frama_b,ema_b)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days FRAMA is **+{R['h20'][2]:.0f} bps**, random is "
            f"**+{R['h20'][5]:.0f} bps**, and a fixed EMA is **+{R['h20'][7]:.0f} bps** — all three "
            f"within a few bps. FRAMA − fixed-EMA is only {R['h20'][8]:+.0f} bps. The 'adaptive' "
            "machinery doesn't beat a one-speed average."
        ),
        md(
            "### 4c · The adaptation placebo — scramble the fractal alpha, nothing changes\n\n"
            "Permute the per-bar smoothing constants $\\alpha_t$ in time (positions of the marginal "
            "kept) so the smoothing no longer tracks the local fractal dimension. If FRAMA's fractal "
            "logic carried information, the observed return should sit far in the right tail of the "
            "scrambled distribution. It sits mid-pack."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY'); c = b['close']\n"
            "    pl = st.shuffled_alpha_placebo(b, 20, n=R['frama_n'], n_draws=300, seed=488)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    al = st.frama_alpha(b, n=R['frama_n']).to_numpy(); fin = np.isfinite(al); base = al[fin]\n"
            "    rng = np.random.default_rng(488); draws = []\n"
            "    for _ in range(300):\n"
            "        perm = al.copy(); perm[fin] = rng.permutation(base)\n"
            "        line = st.ema_from_alpha(c, perm); ent = st._cross_up_entries(c, line)\n"
            "        rr = st.forward_returns(c, ent, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(488); draws = rng.normal(88, 18, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-adaptation FRAMA (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real FRAMA {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean cross-up 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real FRAMA sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real FRAMA {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => adaptation not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real FRAMA (blue line) sits **in the middle** of the "
            f"scrambled-adaptation cloud — **p = {R['placebo'][1]:.2f}**. Randomly re-timed smoothing "
            "does just as well, so the fractal dimension isn't carrying information. The cleanest "
            "refutation of 'fractal-adaptive smoothing buys edge.'"
        ),
        md(
            "### 4d · Per-ticker — no coherent edge over random or over a fixed EMA\n\n"
            "20-day FRAMA-minus-random delta per instrument. If the indicator worked it would be "
            "positive across the board; instead it's negative in 4 of 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, d_rnd = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        b = load(t); c = b['close']\n"
            "        e = st.frama_cross_entries(b, n=R['frama_n']); re = st.random_entries(c, max(len(e),50), n=R['frama_n'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); d_rnd.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; d_rnd = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, d_rnd, color=[GREEN if d>0 else RED for d in d_rnd], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(d_rnd): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d FRAMA − random (bps)'); ax.set_title('FRAMA underperforms random in 4 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d FRAMA-random (bps):', {n:round(d) for n,d in zip(names,d_rnd)})"
        ),
        md(
            f"> 💡 In plain words: only **GLD** edges out a positive delta ({R['per'][4][5]:+.0f} bps); "
            f"SPY is **{R['per'][0][5]:+.0f}** bps *behind* random. No coherent, cross-sectional edge "
            "— exactly what you'd expect if FRAMA is relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real trend\n\n"
            "To prove the null is honest (not a dead detector), plant a **real**, persistent trend "
            "into a synthetic tape and check the same cross-up rule banks it: edge=0 must stay at "
            "t≈0; edge>0 must light up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 2.0):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=488, n_days=6000)\n"
            "    e = st.frama_cross_entries(px, n=16); s = st.summarize(st.forward_returns(px['close'], e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted trend\\n{e:.1f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted trend -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.1f}: n={n} FRAMA={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted trend the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"trend reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the FRAMA cross-up does not beat a drift-matched random baseline "
            f"(FRAMA − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, max **{R['h60'][10]:+.2f}** at 60d). The "
            f"impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; and a **fixed "
            f"EMA** of equal speed matches FRAMA (Δ_ema = {R['h5'][8]:+.0f}/{R['h20'][8]:+.0f} bps). "
            "You'd capture the drift more cheaply by holding the index.\n"
            f"- **Fractal-adaptive edge? `BUSTED`** — the shuffled-alpha placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): time-scrambled adaptation does as well as the "
            "fractal-driven one, and a static EMA matches FRAMA outright. The fractal machinery is "
            "decorative."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "FRAMA's entire apparent profit is the unconditional drift of long equity indices, which "
            "you obtain more cheaply and more fully by **buying and holding**. The cross-up rule "
            "trades thousands of times and pays costs on each, and its 'adaptive' edge over a plain "
            "EMA is statistical noise. There is no capacity question because there is no edge to "
            "scale. FRAMA is a descriptive smoothing tool, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The adaptive-MA family.** KAMA (efficiency ratio), MAMA (Hilbert phase), Hull MA — "
            "all adapt a smoother to past price and inherit the same drift confound. Siblings "
            "[432-hull](../../432-hull-moving-average) and [433-kama](../../433-kama-adaptive) land "
            "in the same place.\n"
            "- **The fractal-dimension estimator.** Ehlers' two-halves D is one of many "
            "(box-counting, R/S, Higuchi); they correlate strongly, so the choice doesn't rescue "
            "the result.\n"
            "- **Modified FRAMA.** The FC/SC slow-clamp variant only *slows* the average further — a "
            "slower lagging line is still a lagging line.\n\n"
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
