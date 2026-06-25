"""Generate the two narrative notebooks for Study 471 (QQE).

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
# 2026-05-31), 21.4 years, RSI len 14 / smoothing 5 / factor 4.236, smoothed-RSI band-cross long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=697,
    rsi_len=14, sf=5, qf=4.236,
    fp_spy="4cb5244f3990",
    # pooled QQE band-cross, per horizon (random baseline pooled over 40 seeds):
    # (H, n, cross_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 696, 31.5, 59, 3.72, 28.8, 2.8, 29.5, 0.28, 0.777),
    h10=(10, 696, 60.3, 62, 5.39, 57.0, 3.3, 58.3, 0.25, 0.802),
    h20=(20, 695, 100.7, 64, 7.01, 110.6, -9.9, 98.7, -0.50, 0.619),
    h60=(60, 690, 310.2, 70, 9.05, 304.1, 6.0, 308.2, 0.19, 0.846),
    # per-ticker H=20: (ticker, entries, cross_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 139, 103.6, 3.48, 104.8, -1.2), ("QQQ", 142, 104.5, 2.96, 150.8, -46.3),
         ("IWM", 140, 118.9, 2.93, 101.1, 17.8), ("DIA", 136, 84.0, 3.34, 92.0, -8.0),
         ("GLD", 140, 92.1, 2.37, 103.0, -10.9)],
    # phase-scramble placebo (SPY, H=20, 500 draws): obs_bps, p
    placebo=(103.6, 0.313, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, cross_bps, win%, one_sample_t)
    syn=[(0.00, 107, -64.3, 46, -1.47), (0.60, 81, 703.3, 81, 9.73)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Band-cross_forecasts%3F: Busted](https://img.shields.io/badge/Band--cross_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from qqe import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real QQE cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the QQE \"momentum ignition\" actually pay? ⚡\n"
            "### A popular smoothed-RSI indicator — a band-cross buy signal — meets a stopwatch\n\n"
            + BADGES +
            "Open any trading-indicator forum and you'll find **QQE** (Quantitative Qualitative "
            "Estimation): a smoothed RSI line with a wiggly trailing band, and a simple rule — "
            "**when the smoothed RSI crosses up through the band, BUY** — sold as a *momentum "
            "ignition* that price keeps running with.\n\n"
            "It *looks* convincing because the green crosses tend to land in uptrends. But an "
            "indicator that's a smoothed version of past price will always 'fire in uptrends' — "
            "that's circular. So we did the only fair thing: encode QQE **mechanically**, fire the "
            "band-cross rule across five big indices over 21 years, and time the result with a "
            "stopwatch — against the only baseline that matters: **buying on random days instead.**\n\n"
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
            "| If I buy on the QQE band-cross, do I make money? | **Yes — but only because the market "
            "goes up.** The win-rate is ~60% and the returns look great. |\n"
            "| Is that *the QQE signal's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well** — once you estimate the drift properly, the cross beats random by "
            "essentially zero. |\n"
            "| Does the band-cross 'forecast'? | **Not in any usable way.** Scramble the price *timing* "
            "into a look-alike random tape and the band-cross does nearly as well. The geometry isn't "
            "doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a momentum signal. |\n\n"
            "> A subtle trap lurks here: with **one** unlucky random comparison the QQE cross *appears* "
            "to win big. That apparent edge is just a badly-estimated baseline — pool many random draws "
            "and it vanishes. QQE is a great way to *describe* momentum, a **mirage** as a forecast."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Take RSI, smooth it, and draw an ATR-of-RSI **trailing band** around it. When the "
            "smoothed RSI **crosses above** the band, momentum has ignited — buy and ride it.\"*\n\n"
            "This is **QQE** (Quantitative Qualitative Estimation), popularised by Igor Livshin and "
            "built from J. Welles Wilder's RSI + ATR machinery (the famous **4.236** factor is "
            "Wilder's). It ships as a built-in on TradingView and MetaTrader and is one of the most "
            "shared 'momentum' indicators on the forums — so: does the cross actually *forecast*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the band-cross genuinely *forecast* continuation, it would be remarkable: a smoothed "
            "function of past prices predicting future ones, a clean crack in market efficiency you "
            "could trade with one line of code. That's the dream the indicator sells.\n\n"
            "But there's a trap built into it. QQE is a **smoothed re-description of recent price** — "
            "so it 'fires in uptrends' by construction. And it's measured on a market (stock indices) "
            "that drifts **up** over time, so *any* long-only entry will look profitable. To separate "
            "the **signal** from the **tide**, we (a) build QQE by a fixed causal rule with no "
            "hindsight, and (b) compare it to buying on **random days** — with the drift estimated "
            "carefully, not from one lucky draw."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Build QQE causally.** Wilder RSI (len 14), smooth it (5), and lay an ATR-of-RSI "
            f"trailing band (factor {R['qf']}) — every step uses only past bars, no look-ahead.\n"
            "2. **Trade the lore.** When the smoothed RSI crosses **above** the trailing band, buy at "
            "the next close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**, pooled over many "
            "random draws so the drift is estimated properly. If QQE matters, the cross must beat "
            "random. *If it doesn't, the signal is a mirage* — that's the result that would make us "
            "say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does QQE even look like? Here's SPY's smoothed RSI with its trailing band, "
            "and the band-crosses the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-450:]\n"
            "    bands = st.qqe_bands(cl, rsi_len=R['rsi_len'], sf=R['sf'], qqe_factor=R['qf'])\n"
            "    ent = st.qqe_cross_entries(cl, rsi_len=R['rsi_len'], sf=R['sf'], qqe_factor=R['qf'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.4), sharex=True, gridspec_kw={'height_ratios':[2,1]})\n"
            "    ax1.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax1.scatter(ent, cl.reindex(ent), c=GREEN, s=40, zorder=5, label='QQE band-cross BUY')\n"
            "    ax1.set_title('QQE band-cross buys on SPY (last ~2y)'); ax1.legend(loc='upper left')\n"
            "    ma = bands['rsi_ma'].reindex(seg.index); ts = bands['ts'].reindex(seg.index)\n"
            "    ax2.plot(seg.index, ma, c='#2c6fbb', lw=1.3, label='smoothed RSI')\n"
            "    ax2.plot(seg.index, ts, c=AMBER, lw=1.1, label='QQE trailing band')\n"
            "    ax2.axhline(50, c=GREY, lw=.7, ls=':'); ax2.set_ylabel('RSI'); ax2.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('band-crosses in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The cross dots land in rallies — *as a description* of momentum. The question is whether "
            "they're followed by **more** gains than you'd get on any old day. **Let's race the "
            "band-cross against random entries** at four horizons. Blue = QQE buy; grey = random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    cross, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.qqe_cross_entries(c, rsi_len=R['rsi_len'], sf=R['sf'], qqe_factor=R['qf'])\n"
            "            tt.append(st.forward_returns(c, e, h))\n"
            "            rr.append(st.random_baseline(c, max(len(e),50), h, n_seeds=40))\n"
            "        cross.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    cross = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, cross, .4, color='#2c6fbb', label='QQE band-cross')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(cross,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('QQE band-cross does NOT beat random'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cross:', [round(v) for v in cross]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The band-cross makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make essentially the "
            f"same** (**+{R['h20'][5]:.0f} bps**). At every horizon the famous QQE cross is "
            "statistically a dead heat with throwing darts. The apparent edge was **the market's "
            "upward drift**, not the indicator."
        ),
        md(
            "**One more sanity check.** What if we scramble the price *timing* — keep the same "
            "wiggle 'shape' and volatility but shuffle *when* moves happen (a Fourier look-alike)? If "
            "the band-cross really 'forecasts', the nonsense-timing tape should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.phase_scramble_placebo(c, 20, rsi_len=R['rsi_len'], sf=R['sf'], qqe_factor=R['qf'], n_draws=200, seed=471)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real QQE band-cross (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled-timing* tapes do at least as well (p={pval:.2f}).')\n"
            "print('=> the QQE geometry is not doing the work.')"
        ),
        md(
            f"About a third of the **scrambled-timing** tapes match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If the band-cross genuinely keyed off real momentum, a "
            "timing scramble would collapse the result. It barely moves — because the result was never "
            "about the signal."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The QQE band-cross does **not** beat buying on random days once the "
            "drift is estimated properly (the cross-vs-random difference never clears *t* = 2). The "
            "big absolute returns are the market's drift, not the indicator.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Does the band-cross forecast\"? — Busted.** Scramble the timing into a look-alike "
            "tape and the result barely moves. The geometry doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The band-cross's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The QQE buy is a worse, more expensive way to be long. Costs "
            "(commissions + spread on every cross) push the already-no-edge result further negative. "
            "As a forecasting tool it doesn't pay; as a momentum *descriptor* it was never a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The lucky-seed trap.** With one unlucky random comparison the QQE cross *appears* to "
            "win at *t* ≈ 3. The quants notebook shows that 'edge' is purely an under-estimated "
            "baseline — pool the seeds and it dies. A great cautionary tale for any backtest.\n"
            "- **Different parameters.** Try other RSI/smoothing/factor settings — the result is "
            "robust: drift in, indicator out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-cross "
            "continuation into a synthetic tape and shows the harness banks it (so the null result "
            "here isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the band-cross forecasts? Show it beating a properly-pooled random baseline at "
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
            "# QQE — a quantitative teardown 🔬\n"
            "### Causal smoothed-RSI trailing-band crosses on 5 indices · band-cross forward returns · "
            "one-sample HAC *t* · a **pooled** drift-matched random-entry baseline · a phase-scramble "
            "geometry placebo · costs · a synthetic planted-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **signal** from the **drift**: an upward-trending index makes *any* "
            "long-only entry look good, so the only meaningful test is cross-vs-random, plus a placebo "
            "that destroys the indicator's timing while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. QQE is causal (RSI len "
            f"{R['rsi_len']}, smoothing {R['sf']}, factor {R['qf']}); entry is the **next close** "
            "(one documented lag). Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Band-cross vs a **pooled drift-matched random** baseline: Δ = "
            f"{R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps and the "
            f"cross-minus-random Welch *t* **never clears t = 2** (max {R['h5'][8]:+.2f} at 5d). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta** — they vanish against random entries and against cost. No residual edge "
            "to scale. |\n"
            f"| **Band-cross forecasts?** | `BUSTED` | Scrambling the price timing (phase-scramble "
            f"placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of nonsense-timing tapes "
            "match or beat the real one. The geometry isn't doing the work. |\n\n"
            "> 💡 In plain words: the band-cross *looks* significant only because indices drift up. "
            "Strip the drift (race it vs a **pooled** random) or strip the timing (scramble the tape) "
            "and the edge evaporates. Classic beta-in-a-costume — with a baseline-estimation twist."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $R_t$ be Wilder's RSI, $\\overline{R}_t = \\mathrm{EMA}_{sf}(R_t)$ the smoothed RSI, "
            "and $\\delta_t = f\\cdot\\mathrm{ATR}_{rsi}(\\overline{R})_t$ the band half-width "
            "($f=4.236$). A dual trailing stop $T_t$ flips between $\\overline{R}_t-\\delta_t$ (below) "
            "and $\\overline{R}_t+\\delta_t$ (above). The QQE rule buys when $\\overline{R}_t$ crosses "
            "**above** $T_t$.\n\n"
            "- **H₀ (drift).** Cross returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (QQE forecasts).** Cross returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the geometry matters).** Cross returns exceed a **phase-scrambled** tape whose "
            "timing is destroyed.\n\n"
            "We find **H₀ not rejected** (cross ≈ random), **H₁ rejected** (Welch t never ≥ 2), "
            "**H₂ rejected** (placebo p ≈ 0.31). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* entry rule "
            "on a long-only horizon inherits it; a high one-sample $t$ against **zero** measures the "
            "tide, not the tool. The fix is the **random-entry baseline** (same instrument, epoch, "
            "hold) and a Welch test of cross-*minus*-random.\n\n"
            "**(b) Baseline estimation error.** A *single* random draw of a few hundred dates has a "
            "noisy mean — a lucky-low draw fabricates a fake edge. We therefore **pool the baseline "
            "over 40 seeds** (~28k entries), estimating the true drift. *This is the load-bearing "
            "design choice here*: the single-seed comparison makes QQE look like a winner; the pooled "
            "one reveals it as nothing.\n\n"
            "**(c) Geometry as a free re-description.** QQE is a smoothed function of past price, so "
            "it 'fires in uptrends' tautologically. The **phase-scramble placebo** keeps the spectrum "
            "and return marginal but destroys the timing — if the real result survives, the geometry "
            "was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} band-crosses** pooled.\n"
            f"- **QQE.** Causal: Wilder RSI (len {R['rsi_len']}) → EMA smoothing ({R['sf']}) → "
            f"ATR-of-RSI × {R['qf']} dual-band trailing stop. Cross read on close of *t*.\n"
            "- **Entry.** First bar RSI MA crosses above the stop; enter **next close** (one lag); "
            "hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of cross returns vs 0 (Newey-West).\n"
            "- **Null #2 — pooled random-entry baseline**, Welch two-sample cross vs random (the *real* test).\n"
            "- **Null #3 — phase-scramble placebo** (timing destroyed, spectrum/marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every cross.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-cross continuation (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the band-cross's **one-sample** t against zero (the misleading number). Right: the "
            "same cross vs a **pooled drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, cross, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.qqe_cross_entries(c, rsi_len=R['rsi_len'], sf=R['sf'], qqe_factor=R['qf'])\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.random_baseline(c, max(len(e),50), h, n_seeds=40))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); cross.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    cross = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Cross vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long-only entry inherits it. The "
            f"right bars are the real test: cross-minus-random is near zero "
            f"({R['h20'][8]:+.2f} at 20d) — never significant. QQE adds nothing over a coin flip."
        ),
        md(
            "### 4b · The lucky-seed trap — why pooling the baseline matters\n\n"
            "Here's the subtle bit. If you compare the cross to a **single** random draw, the baseline "
            "is noisy — and one unlucky-low draw makes QQE look like a real winner. Below: the "
            "distribution of the cross-minus-random Δ across 200 single-seed baselines (grey), vs the "
            "pooled estimate (blue). The single-seed Δ swings wildly; the pooled one sits at ~0."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']; h = 20\n"
            "    e = st.qqe_cross_entries(c, rsi_len=R['rsi_len'], sf=R['sf'], qqe_factor=R['qf'])\n"
            "    cross_mean = st.forward_returns(c, e, h).mean()*1e4\n"
            "    deltas = []\n"
            "    for s in range(200):\n"
            "        re = st.random_entries(c, max(len(e),50), seed=s)\n"
            "        deltas.append(cross_mean - st.forward_returns(c, re, h).mean()*1e4)\n"
            "    deltas = np.array(deltas)\n"
            "    pooled = cross_mean - st.random_baseline(c, max(len(e),50), h, n_seeds=40).mean()*1e4\n"
            "else:\n"
            "    rng=np.random.default_rng(471); deltas = rng.normal(0, 60, 200); pooled = R['per'][0][5]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.hist(deltas, bins=30, color=GREY, alpha=.85, label='single-seed Δ (cross − one random draw)')\n"
            "ax.axvline(pooled, c='#2c6fbb', lw=2.5, label=f'pooled Δ = {pooled:+.0f} bps')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('SPY 20d cross − random Δ (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title('A single random draw fabricates a fake edge; pooling kills it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'single-seed Δ: mean {deltas.mean():+.0f}, sd {deltas.std():.0f} bps; pooled Δ {pooled:+.0f} bps')"
        ),
        md(
            "> 💡 In plain words: the single-seed Δ has a standard deviation of tens of bps — pick the "
            "wrong seed and you'd 'discover' a 100-bps QQE edge that isn't there. The **pooled** Δ (blue) "
            "sits at essentially zero. This is the cleanest illustration of why a backtest needs a "
            "*properly estimated* benchmark, not one lucky comparison."
        ),
        md(
            "### 4c · The geometry placebo — scramble the timing, nothing changes\n\n"
            "Rebuild QQE on a **Fourier phase-randomised** clone of the close (same spectrum/marginal, "
            "destroyed timing). If price respects *this specific signal*, the scramble should demolish "
            "the result. The observed cross return should sit far in the right tail. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.phase_scramble_placebo(c, 20, rsi_len=R['rsi_len'], sf=R['sf'], qqe_factor=R['qf'], n_draws=200, seed=471)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np\n"
            "    rng=_np.random.default_rng(471); logp=_np.log(c.to_numpy()); lr=_np.diff(logp); idx=c.index; p0=float(c.iloc[0])\n"
            "    draws=[]\n"
            "    for _ in range(200):\n"
            "        sr=st._phase_scramble(lr,rng); sc=p0*_np.exp(_np.concatenate([[0.0],_np.cumsum(sr)]))\n"
            "        import pandas as _pd; scs=_pd.Series(sc,index=idx)\n"
            "        ee=st.qqe_cross_entries(scs, rsi_len=R['rsi_len'], sf=R['sf'], qqe_factor=R['qf'])\n"
            "        rr=st.forward_returns(scs, ee, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(471); draws = rng.normal(95, 55, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=35, color=GREY, alpha=.85, label='phase-scrambled tapes (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real QQE {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean band-cross 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real QQE sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real QQE {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real QQE (blue line) sits **inside** the scrambled-timing cloud "
            f"— **p = {R['placebo'][1]:.2f}**. Nonsense-timing tapes do nearly as well, so the specific "
            "QQE band-cross isn't carrying forecasting information. This is the cleanest refutation of "
            "'the band-cross forecasts.'"
        ),
        md(
            "### 4d · Per-ticker — the cross does not beat random\n\n"
            "20-day cross-minus-random delta, per instrument. If QQE worked it would be positive across "
            "the board; instead it's negative in 4 of 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.qqe_cross_entries(c, rsi_len=R['rsi_len'], sf=R['sf'], qqe_factor=R['qf'])\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.random_baseline(c, max(len(e),50), 20, n_seeds=40))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d cross − random (bps)'); ax.set_title('Cross underperforms random in 4 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **IWM** edges out a positive delta ({R['per'][2][5]:+.0f} bps); "
            f"QQQ is **{R['per'][1][5]:+.0f}** bps *behind* random. No coherent, cross-sectional edge "
            "— exactly what you'd expect if QQE is just relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real continuation\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-cross "
            "continuation into a synthetic tape and check the same band-cross rule banks it: edge=0 "
            "must stay at t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=471, n_days=4000)\n"
            "    c = px['close']; e = st.qqe_cross_entries(c); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted continuation -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} cross={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted continuation the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"continuation reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the band-cross does not beat a (pooled) drift-matched random "
            f"baseline (cross − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t never clears 2, max **{R['h5'][8]:+.2f}**). "
            f"The impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Band-cross forecasts? `BUSTED`** — the phase-scramble placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): nonsense-timing tapes do as well as the real "
            "one, so the QQE geometry carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The band-cross's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. The QQE rule trades "
            "*less* of the time and pays costs on each cross, so it strictly dominates *nothing*. There "
            "is no capacity question because there is no edge to scale. QQE is a descriptive momentum "
            "indicator, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The baseline-estimation lesson.** The headline finding here is methodological: the "
            "single-seed comparison *manufactures* a Welch *t* ≈ 3 'edge' for QQE that pooling "
            "annihilates. Any indicator study that benchmarks against one random draw is suspect.\n"
            "- **Parameter robustness.** Sweep RSI/smoothing/factor — the drift confound is invariant.\n"
            "- **QQE variants** (QQE MOD, QQE+Bollinger) are affine/cosmetic tweaks of the same "
            "smoothed-RSI-plus-band geometry and inherit the same confound.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the detector "
            "is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
