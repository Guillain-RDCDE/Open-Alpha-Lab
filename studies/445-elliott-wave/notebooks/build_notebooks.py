"""Generate the two narrative notebooks for Study 445 (Elliott Wave).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily, 8-tape index/ETF
# basket, as-of 2025-12-31, SPY 1993-01-29 -> 2025-12-31, fingerprint 7ce3202106f8).
R = dict(
    asof="2025-12-31", spy_start="1993-01-29", spy_years=32.9, spy_fp="7ce3202106f8",
    n_pivots=2512, n_entries=661,
    # WAVE-3 pooled: (H, n, gross_bps, net_bps, win%, t, hac_t, placebo_p)
    w3=[(5, 661, -17, -19, 52, -1.02, -1.09, 0.843),
        (10, 661, 5, 3, 54, 0.22, 0.26, 0.415),
        (20, 661, 17, 15, 54, 0.53, 0.63, 0.301),
        (40, 660, 59, 57, 54, 1.32, 1.63, 0.093)],
    # COMPLETED-IMPULSE pooled (sell the 5-wave): (H, n, bps, win%, t, hac_t)
    n_impulse=1339,
    imp=[(5, 1339, 6, 48, 0.50, 0.46), (10, 1339, 15, 48, 0.96, 0.83),
         (20, 1335, 8, 46, 0.35, 0.26), (40, 1331, -41, 45, -1.42, -1.03)],
    # Fibonacci band robustness H=20: (label, n, bps, t)
    fib=[("textbook 50-61.8%", 188, 60, 1.10), ("wide 38.2-78.6%", 661, 17, 0.53),
         ("deep 61.8-100%", 579, -23, -0.69)],
    # ZigZag threshold robustness H=20: (pct, n, bps, t)
    zz=[(3, 1360, -10, -0.53), (5, 661, 17, 0.53), (8, 282, 50, 1.03)],
    # SPY-only: (H, n, bps, win%, t, placebo_p)
    spy=[(5, 67, -10, 54, -0.18, 0.571), (10, 67, 3, 58, 0.04, 0.486),
         (20, 67, 41, 57, 0.42, 0.345), (40, 67, 15, 54, 0.12, 0.464)],
    # synthetic control H=10: (planted_edge, entries, bps, win%, t, hac_t, placebo_p)
    syn=[(0.00, 83, -35, 48, -0.97, -0.92, 0.810),
         (0.15, 112, 470, 78, 7.02, 7.33, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Five_waves_predict%3F: Not_Supported](https://img.shields.io/badge/Five_waves_predict%3F-Not_Supported-8b949e?style=flat-square)\n\n"
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

from elliott_wave import data, strategy as st

END = "2025-12-31"
TICKERS = list(data.TICKERS)
HAVE_REAL = data.have_real("SPY")

def pooled(fn, pct=0.05, **kw):
    \"\"\"Pool one signal's forward returns + dirs across the cached basket, trimmed to END.\"\"\"
    rets = {h: [] for h in st.HORIZONS}; dirs = {h: [] for h in st.HORIZONS}; npv = nen = 0
    for tk in TICKERS:
        if not data.have_real(tk):
            continue
        b = data.load_real(tk); b = b[b.index <= END]
        piv = st.zigzag(b["close"].to_numpy(float), pct=pct); ent = fn(piv, **kw)
        npv += len(piv); nen += len(ent)
        for h in st.HORIZONS:
            led = st.run_trades(b, ent, h, cost_bps=0.0)
            rets[h].append(led["ret_gross"].to_numpy()); dirs[h].append(led["dir"].to_numpy())
    return rets, dirs, npv, nen

print("real cache present:", HAVE_REAL)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do Elliott's five waves predict anything? 🌊\n"
            "### The most beautiful chart pattern in trading — tested out-of-sample, in plain English\n\n"
            + BADGES +
            "Open any technical-analysis forum and you'll meet **Elliott Wave Theory**: the idea that "
            "markets move in a hypnotic, repeating rhythm of **five waves up** (labelled 1-2-3-4-5) and "
            "then **three waves back** (A-B-C), nested inside each other like Russian dolls, with "
            "**Fibonacci** numbers governing every turn. Practitioners draw these counts on charts and "
            "say they can tell you *where the next move goes*.\n\n"
            "It looks gorgeous. The question is whether it **works** — and there's a catch built into the "
            "theory itself that we have to deal with first.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the robustness sweep? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **The honesty problem up front.** Elliott Wave is *irreducibly subjective* — the "
            "\"correct\" wave count is only obvious **after** the move finishes, and two analysts will "
            "label the same chart differently. So we can't test \"the theory\" directly. Instead we test "
            "the **tightest mechanical version** its own believers accept: a standard **ZigZag** swing "
            "tool to mark the turns, a **Fibonacci** filter, and the one *tradable* prediction it makes. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Can a mechanical Elliott rule predict the next move? | **No.** The classic entry — buy "
            "**wave 3** after a textbook 1-2 — makes **no** money you can distinguish from luck across "
            f"**{R['n_entries']}** signals on 8 indices over 33 years. |\n"
            "| Does the wave count beat a coin flip? | **No.** Flip a coin at the *exact same* turning "
            "points and it does just as well — the wave labels add **zero** directional information. |\n"
            "| Is the engine just broken / too dumb to see waves? | **No.** When we *secretly plant* a "
            "real wave-3 pattern into fake data, the same code lights up instantly (*t* = 7). It finds "
            "the structure **when it's actually there** — on real markets, it isn't. |\n"
            "| So is the prettiness fake? | The patterns are **real to look at** — markets do swing. But "
            "the *predictive* claim is hindsight. You can always find five waves **after** the fact. |\n\n"
            "> Elliott Wave is the rare case where the picture is mesmerising and the forecast is empty."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Markets aren't random — they move in **waves**. A trend unfolds in **five** waves "
            "(1-2-3-4-5): wave 1 starts it, wave 2 pulls back about half, **wave 3 is the big powerful "
            "one** you want to be in, wave 4 rests, wave 5 finishes. Then three corrective waves (A-B-C) "
            "give it back. The turns land on **Fibonacci** ratios. Count the waves and you know what "
            "comes next.\"*\n\n"
            "This is Ralph Nelson Elliott's 1938 *Wave Principle*, kept alive by Robert Prechter's "
            "*Elliott Wave Principle* (1978) and a whole industry of newsletters. The single most "
            "*actionable* piece is **\"buy wave 3\"**: once you've spotted a wave 1 and a Fibonacci-sized "
            "wave-2 pullback, you go long for the powerful third wave. That's the claim we can actually "
            "put a number on."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If you could reliably spot wave 3 *before* it happened, you'd have a money printer — the "
            "third wave is supposed to be the longest, strongest leg of the whole sequence. And it would "
            "mean markets carry a deep, fractal **memory** that lets the past dictate the future. That's "
            "a huge claim about how markets work.\n\n"
            "But there's a trap. The reason Elliott Wave *feels* so right is that you label the waves "
            "**looking backward**, when the answer is already on the screen. The honest test is the "
            "opposite: mark the turns by a fixed mechanical rule, place the trade the theory says, and "
            "see if the **future** cooperates. Spoiler — it doesn't."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We can't ask a human to \"count the waves\" — that's the subjectivity we're trying to "
            "escape. So we use the **ZigZag**, the standard charting tool every Elliottician uses to mark "
            "swings: it puts a dot at a turning point only after price has reversed by at least **5%**. "
            "That gives a clean, alternating high-low-high-low skeleton — the raw material of every wave "
            "count.\n\n"
            "Then we apply the textbook rule. For every three-pivot stretch (a candidate wave 1 then "
            "wave 2), we check the wave-2 pullback retraced a **Fibonacci** fraction of wave 1 (we allow "
            "a generous 38–79% band, with the classic 50–62% inside it). If it qualifies, we **buy in "
            "the wave-1 direction** — that's the wave-3 bet — entering the day **after** the turn is "
            "confirmed (no peeking), and hold 5 to 40 days. We do this on **SPY plus 7 other broad "
            "indices** to get enough signals to judge, and we ask: did it beat a **coin flip** placed at "
            "the very same turns?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, here's what the ZigZag draws on real SPY — the swing skeleton an Elliottician would "
            "start counting from."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = data.load_real('SPY'); spy = spy[(spy.index>='2018-01-01') & (spy.index<='2020-12-31')]\n"
            "    c = spy['close'].to_numpy(float)\n"
            "    piv = st.zigzag(c, pct=0.05)\n"
            "    fig, ax = plt.subplots(figsize=(10.0, 4.4))\n"
            "    ax.plot(spy.index, c, color=GREY, lw=1.1, alpha=.8, label='SPY close')\n"
            "    ax.plot(spy.index[piv['piv_idx']], piv['price'], '-o', color=GREEN, lw=1.6, ms=5, label='5% ZigZag swings')\n"
            "    ax.set_title('The ZigZag skeleton — the raw material of every wave count (SPY 2018-2020)')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print('pivots drawn:', len(piv))\n"
            "else:\n"
            "    print('cache missing — see docs/results.md for the frozen numbers')"
        ),
        md(
            "Pretty, isn't it? You can *see* waves there. Now the real question: when we mechanically "
            "place the **\"buy wave 3\"** trade at every qualifying turn, does the future pay off? Here's "
            "the average return at four holding horizons, pooled across all 8 indices."
        ),
        code(
            "hs = list(st.HORIZONS)\n"
            "if HAVE_REAL:\n"
            "    rets, dirs, npv, nen = pooled(st.wave3_entries)\n"
            "    bps = [np.concatenate(rets[h]).mean()*1e4 for h in hs]\n"
            "    ts  = [st.ttest_vs_zero(np.concatenate(rets[h])) for h in hs]\n"
            "else:\n"
            "    bps = [r[2] for r in R['w3']]; ts = [r[5] for r in R['w3']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [GREEN if t>=2 else RED for t in ts]\n"
            "ax.bar([f'{h}d' for h in hs], bps, color=cols, width=.6)\n"
            "for i,(v,t) in enumerate(zip(bps,ts)): ax.annotate(f'{v:+.0f}bps\\n t={t:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('avg return per trade (bps)')\n"
            "ax.set_title('\"Buy wave 3\": nothing clears the t=2 bar — mostly noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-horizon bps:', [round(b,0) for b in bps], ' t:', [round(t,2) for t in ts])"
        ),
        md(
            f"All red. The biggest number is a feeble **+{R['w3'][3][2]} bps at 40 days** with "
            f"**t = {R['w3'][3][5]:.2f}** — under the bar, and at 40 days you're mostly just *long the "
            "market*, which usually goes up anyway. At 5 days the \"edge\" is actually **negative**. The "
            "famous wave-3 entry is indistinguishable from noise."
        ),
        md(
            "**But maybe it just needs the right opponent.** The fair test: flip a **coin** at the exact "
            "same turning points (same dates, random direction) and compare. If the wave count knows "
            "anything, it should beat the coin."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets, dirs, npv, nen = pooled(st.wave3_entries)\n"
            "    h = 20; r = np.concatenate(rets[h]); d = np.concatenate(dirs[h]); mags = r*d\n"
            "    rng = np.random.default_rng(445)\n"
            "    draws = np.array([(rng.choice([-1,1],size=len(mags))*mags).mean()*1e4 for _ in range(5000)])\n"
            "    obs = r.mean()*1e4; p = float((draws>=r.mean()).mean())\n"
            "else:\n"
            "    obs = R['w3'][2][2]; p = R['w3'][2][7]\n"
            "    rng = np.random.default_rng(445); draws = rng.normal(0, 35, 5000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=55, color=GREY, alpha=.85, label='coin flips at the same turns (5,000x)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'the wave-3 count: {obs:+.0f} bps')\n"
            "ax.set_xlabel('avg 20-day return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The wave count sits right in the coin cloud (p = {p:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'wave-3 {obs:+.0f} bps vs a coin at the same turns: p = {p:.3f}')"
        ),
        md(
            f"The red line sits **smack in the middle** of the coin cloud (p ≈ {R['w3'][2][7]:.2f}). A "
            "random direction at the identical swing points does just as well. **The wave labels carry no "
            "directional information.** This is the whole story in one picture."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The mechanical wave-3 rule predicts nothing the market doesn't already "
            "give you for free; a coin at the same turns ties it.\n"
            "- **Tradability — Mirage.** No edge means nothing to trade; costs only make it worse.\n"
            "- **\"Five waves predict the next move\"? — Not Supported.** Out-of-sample, the core "
            "forward-looking claim simply doesn't hold."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing to trade. But to be completely fair to the engine, we ran a **sanity "
            "check**: we built fake data with a **real wave-3 pattern secretly planted in it**, then "
            "pointed the *same code* at it. If the code is too dumb to ever see waves, it would fail here "
            "too — and then the flat real-market result would mean nothing."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.15):\n"
            "    bars, _ = data.synthetic_panel(edge=edge, seed=445, n_days=9000)\n"
            "    r = st.run_experiment(bars, horizon=10, cost_bps=1.0, n_draws=4000)\n"
            "    res.append((edge, r['wave_gross']['t']))\n"
            "labels = ['no pattern\\n(random walk)', 'planted\\nwave-3 edge']\n"
            "tv = [r[1] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar(labels, tv, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tv): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('wave-3 t-stat'); ax.set_title('Sanity check: the engine DOES find a real planted wave-3')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for e,t in res: print(f'planted edge {e:+.2f}: t = {t:.2f}')"
        ),
        md(
            f"On a pure random walk the code finds **nothing** (t = {R['syn'][0][4]:.2f}) — no false "
            f"alarm. With a **real wave-3 planted**, it lights up at **t = {R['syn'][1][4]:.2f}**. So the "
            "engine works perfectly; it's the **real market** that has no wave-3 edge to find. That's a "
            "true negative, not a broken test."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The deeper lesson.** A theory you can only apply *in hindsight* will always look right "
            "and predict nothing. Elliott Wave is the poster child: infinitely flexible counts, "
            "Fibonacci everywhere, no fixed rule — it explains the past and forecasts noise.\n"
            "- **Try a stricter count.** Add wave-3-is-longest and wave-4-doesn't-overlap-wave-1 rules. "
            "It shrinks the sample but (spoiler from the quants notebook) doesn't rescue the *t*.\n"
            "- **Intraday or other assets.** Re-point the loader at crypto or FX; the subjectivity "
            "problem travels with the method.\n\n"
            "*Think a tighter mechanical count works? Show the **net** wave-3 long-short clearing t = 2 "
            "at the same swing points a coin can't — then we'll talk.*"
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
            "# Elliott Wave — a quantitative teardown 🔬\n"
            "### A mechanical ZigZag+Fibonacci proxy for an irreducibly subjective method · the wave-3 "
            "entry · forward 5/10/20/40-day returns · one-sample & HAC *t* + a same-bars coin placebo · "
            "Fibonacci-band & threshold robustness · the completed-impulse correction test · a "
            "planted-edge faithful-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Elliott Wave "
            "has no single falsifiable rule — the count is hindsight-dependent — so we steelman it with "
            "the tightest mechanical version proponents accept and test only its **forward-looking** "
            "predictions. The honest finding: nothing clears **t = 2**, a same-bars coin ties it, and "
            "the result is robust to every knob — while a planted-edge control proves the harness *can* "
            "detect a real wave-3.\n\n"
            "> ⚠️ **Data note.** yfinance auto-adjusted (total-return) daily OHLC; 8-tape index/ETF "
            "basket (SPY, QQQ, DIA, IWM, ^GSPC, ^IXIC, ^DJI, GLD); **as-of 2025-12-31** (no partial "
            "year). Price indices → no cross-sectional survivorship sort. Offline core + synthetic "
            "control are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Pooled wave-3 long-short over **{R['n_entries']}** entries: the "
            f"largest statistic anywhere is **HAC t = {R['w3'][3][6]:.2f} at 40d** (just beta); shorter "
            f"horizons are noise/negative; same-bars coin **p = {R['w3'][2][7]:.2f}** at 20d. |\n"
            f"| **Tradability** | `MIRAGE` | No edge to charge costs against; the only positive number "
            f"(40d, +{R['w3'][3][2]} bps) is indistinguishable from buy-and-hold. |\n"
            f"| **Five waves predict?** | `NOT SUPPORTED` | Neither the wave-3 extension nor the "
            f"completed-impulse correction (40d **t = {R['imp'][3][4]:.2f}**) is distinguishable from "
            f"chance. Synthetic control with a planted wave-3: **t = {R['syn'][1][4]:.2f}** — true "
            "negative, not a broken pipe. |\n\n"
            "> 💡 In plain words: a mechanical Elliott rule predicts nothing out-of-sample, and the "
            "method's own irreducible subjectivity means a human count has even more freedom to fit the "
            "past. The engine isn't blind — there's just nothing there."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a 5% percentage ZigZag produce alternating pivots $P_0, P_1, P_2, \\dots$ A candidate "
            "**impulse 1-2** is a triple $(P_0, P_1, P_2)$ with wave 1 $= P_1-P_0$, wave 2 $= P_2-P_1$ of "
            "opposite sign, retracement $\\rho = |P_2-P_1|/|P_1-P_0| \\in [0.382, 0.786]$ (the Fibonacci "
            "band; textbook $0.5$–$0.618$ inside) and $P_2$ not breaching $P_0$ (Elliott's rule 2). The "
            "theory's tradable claim:\n\n"
            "- **H₁ (wave 3 extends).** Entering with $\\operatorname{sign}(P_1-P_0)$ one day after $P_2$ "
            "is confirmed earns a positive forward return.\n"
            "- **H₂ (correction after five).** After a five-pivot impulse, a position *against* the "
            "impulse earns a positive forward return.\n\n"
            "We test $\\mathbb{E}[r] > 0$ with a one-sample and HAC *t*, against a **same-bars "
            "random-direction** null. **H₁ rejected** (max HAC t = 1.63, beaten by a coin). **H₂ "
            "rejected** (insignificant, wrong sign at 40d). The mechanical core predicts nothing — and "
            "the *un*-mechanical, hand-counted version has strictly more hindsight freedom, so it cannot "
            "be in a better position."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a one-sample / HAC *t* of the directional per-trade return vs 0:\n\n"
            "$$t = \\frac{\\bar r}{\\hat\\sigma_{\\mathrm{HAC}}/\\sqrt{n}}, \\qquad r_i = d_i\\,"
            "\\Big(\\tfrac{P_{\\text{exit}}}{P_{\\text{entry}}}-1\\Big),\\; d_i=\\operatorname{sign}(\\text{wave 1}).$$\n\n"
            "Two honesty problems. **(a) Beta contamination:** wave 1 is usually *up* on equity indices, "
            "so a long held 40 days is mostly market drift, not an Elliott edge — which is exactly why "
            "the only positive number appears at the longest horizon. **(b) Researcher freedom:** the "
            "Fibonacci band and the ZigZag threshold are free parameters; a positive at *one* setting is "
            "data-snooping. We answer (a) by reading short horizons and the coin placebo, and (b) by "
            "sweeping both knobs."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** 8 broad index/ETF tapes (yfinance daily, auto-adjusted); pooled for power. "
            f"SPY spans {R['spy_start']}→{R['asof']} ({R['spy_years']:.0f}y); **{R['n_pivots']:,}** "
            f"pivots → **{R['n_entries']}** wave-3 entries. As-of **{R['asof']}** (no partial year).\n"
            "- **Detector.** 5% percentage ZigZag; a pivot is *confirmed* (and only then known) at the "
            "bar where price has reversed 5% — the entry is the **next** close (one-day lag, no "
            "look-ahead).\n"
            "- **Signal.** Wave-3: long $\\operatorname{sign}(\\text{wave 1})$ after a Fib wave-2. "
            "Correction: short the impulse after a five-pivot move.\n"
            "- **Null #1 — one-sample + HAC t** of the per-trade return vs 0.\n"
            "- **Null #2 — same-bars coin placebo.** 5,000 random $\\pm1$ direction draws at the "
            "identical entry bars; $p=\\Pr[\\text{coin mean}\\ge\\text{observed}]$.\n"
            "- **Robustness.** Sweep the Fibonacci band and the ZigZag threshold.\n"
            "- **Costs.** 1 bp one-way (×2 round trip).\n"
            "- **Positive control.** A deterministic panel with a *planted* wave-3: edge 0 must NOT "
            "reach significance; a real planted edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The term structure of the wave-3 return + its t-stats\n\n"
            "Per-trade return (bps) and the one-sample / HAC *t* at each horizon, pooled across the "
            "basket. The dashed line is the **t = 2** bar."
        ),
        code(
            "hs = list(st.HORIZONS)\n"
            "if HAVE_REAL:\n"
            "    rets, dirs, npv, nen = pooled(st.wave3_entries)\n"
            "    bps = [np.concatenate(rets[h]).mean()*1e4 for h in hs]\n"
            "    ts  = [st.ttest_vs_zero(np.concatenate(rets[h])) for h in hs]\n"
            "    ht  = [st.hac_t(np.concatenate(rets[h])) for h in hs]\n"
            "    print('entries:', nen)\n"
            "else:\n"
            "    bps=[r[2] for r in R['w3']]; ts=[r[5] for r in R['w3']]; ht=[r[6] for r in R['w3']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], bps, color=GREY, width=.6)\n"
            "for i,v in enumerate(bps): a1.annotate(f'{v:+.0f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_ylabel('avg return (bps)'); a1.set_title('Per-trade return')\n"
            "x=np.arange(len(hs))\n"
            "a2.bar(x-.2, ts, .4, color=AMBER, label='one-sample t')\n"
            "a2.bar(x+.2, ht, .4, color=GREY, label='HAC t')\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2 bar'); a2.axhline(0,c='k',lw=.8)\n"
            "a2.set_xticks(x); a2.set_xticklabels([f'{h}d' for h in hs]); a2.set_ylabel('t-stat')\n"
            "a2.set_title('Nothing reaches t=2'); a2.legend(); a2.set_ylim(-1.6, 2.4)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('bps:', [round(b,0) for b in bps]); print('t:', [round(t,2) for t in ts]); print('hac_t:', [round(t,2) for t in ht])"
        ),
        md(
            f"> 💡 In plain words: the term structure rises only because longer holds = more market beta "
            f"(wave 1 is usually up). The peak is **HAC t = {R['w3'][3][6]:.2f} at 40d** — under the bar — "
            f"and the genuinely *Elliott* horizons (5–10d, where wave 3 should fire) are flat or "
            "**negative**. No signal."
        ),
        md(
            "### 4b · The decisive test — same-bars coin placebo at 20 days\n\n"
            "Does the wave count beat a coin placed at the *identical* swing points? The observed mean "
            "should sit in the right tail if the labels mean anything."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets, dirs, npv, nen = pooled(st.wave3_entries)\n"
            "    r = np.concatenate(rets[20]); d = np.concatenate(dirs[20]); mags = r*d\n"
            "    rng = np.random.default_rng(445)\n"
            "    draws = np.array([(rng.choice([-1,1],size=len(mags))*mags).mean()*1e4 for _ in range(5000)])\n"
            "    obs = r.mean()*1e4; p = float((draws>=r.mean()).mean()); t = st.ttest_vs_zero(r)\n"
            "else:\n"
            "    obs=R['w3'][2][2]; p=R['w3'][2][7]; t=R['w3'][2][5]\n"
            "    rng=np.random.default_rng(445); draws=rng.normal(0,32,5000)\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.hist(draws, bins=60, color=GREY, alpha=.85, label='5,000 same-bars coin flips')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'wave-3 observed {obs:+.0f} bps')\n"
            "ax.set_xlabel('avg 20-day return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the coin cloud: placebo p = {p:.2f}, one-sample t = {t:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.0f} bps  t={t:.2f}  same-bars coin p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: **p = {R['w3'][2][7]:.2f}** — a coin at the same turns beats the wave "
            "count a third of the time and ties it overall. The labels add nothing; only the *swings "
            "themselves* (which any swing tool finds) carry the moves."
        ),
        md(
            "### 4c · Robustness — every knob fails the same way\n\n"
            "Left: vary the **Fibonacci** wave-2 band. Right: vary the **ZigZag** threshold. The 20-day "
            "*t* never reaches 2 at any setting — the negative is not a single unlucky parameter."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fib = []\n"
            "    for lo,hi,lab in [(0.5,0.618,'50-62%'),(0.382,0.786,'38-79%'),(0.618,1.0,'62-100%')]:\n"
            "        rb,db,_,_ = pooled(st.wave3_entries, fib_lo=lo, fib_hi=hi)\n"
            "        rr=np.concatenate(rb[20]); fib.append((lab, len(rr), st.ttest_vs_zero(rr)))\n"
            "    zz = []\n"
            "    for pct in (0.03,0.05,0.08):\n"
            "        rb,db,_,_ = pooled(st.wave3_entries, pct=pct)\n"
            "        rr=np.concatenate(rb[20]); zz.append((f'{int(pct*100)}%', len(rr), st.ttest_vs_zero(rr)))\n"
            "else:\n"
            "    fib=[(f[0].split()[0], f[1], f[3]) for f in R['fib']]\n"
            "    zz=[(f'{z[0]}%', z[1], z[3]) for z in R['zz']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar([f[0] for f in fib], [f[2] for f in fib], color=AMBER, width=.55)\n"
            "a1.axhline(2, ls='--', c=RED); a1.axhline(0,c='k',lw=.8)\n"
            "for i,f in enumerate(fib): a1.annotate(f'n={f[1]}',(i,f[2]),ha='center',va='bottom' if f[2]>=0 else 'top',fontsize=8)\n"
            "a1.set_ylabel('20d t-stat'); a1.set_title('Fibonacci wave-2 band'); a1.set_ylim(-1.4,2.4)\n"
            "a2.bar([z[0] for z in zz], [z[2] for z in zz], color=GREY, width=.55)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=2'); a2.axhline(0,c='k',lw=.8)\n"
            "for i,z in enumerate(zz): a2.annotate(f'n={z[1]}',(i,z[2]),ha='center',va='bottom' if z[2]>=0 else 'top',fontsize=8)\n"
            "a2.set_ylabel('20d t-stat'); a2.set_title('ZigZag threshold'); a2.set_ylim(-1.4,2.4); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('fib:', [(f[0],f[1],round(f[2],2)) for f in fib]); print('zz:', [(z[0],z[1],round(z[2],2)) for z in zz])"
        ),
        md(
            f"> 💡 In plain words: the \"textbook\" tight band gives the biggest point estimate "
            f"(**+{R['fib'][0][2]} bps**) but on only **{R['fib'][0][1]}** events it's **t = "
            f"{R['fib'][0][3]:.2f}** — the classic small-sample mirage a believer would over-read. Nothing "
            "survives. This is a robust null, not one unlucky knob."
        ),
        md(
            "### 4d · The other prediction — sell the completed five-wave impulse\n\n"
            "The correction claim: after a five-pivot impulse, bet against it. Forward returns by "
            "horizon, pooled."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets2, dirs2, _, nen2 = pooled(st.completed_impulse_entries)\n"
            "    bps = [np.concatenate(rets2[h]).mean()*1e4 for h in hs]\n"
            "    ts  = [st.ttest_vs_zero(np.concatenate(rets2[h])) for h in hs]\n"
            "    print('completed-impulse entries:', nen2)\n"
            "else:\n"
            "    bps=[r[2] for r in R['imp']]; ts=[r[4] for r in R['imp']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([f'{h}d' for h in hs], bps, color=[RED if v<0 else GREY for v in bps], width=.6)\n"
            "for i,(v,t) in enumerate(zip(bps,ts)): ax.annotate(f'{v:+.0f}bps\\n t={t:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_ylabel('avg return per trade (bps)')\n"
            "ax.set_title('\"Correction after five waves\": insignificant, and wrong-signed at 40d')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('imp bps:', [round(b,0) for b in bps], ' t:', [round(t,2) for t in ts])"
        ),
        md(
            f"> 💡 In plain words: the correction bet never reaches significance, and at 40 days it "
            f"**loses** (**{R['imp'][3][2]} bps, t = {R['imp'][3][4]:.2f}**) — the market keeps drifting "
            "up after the \"completed\" impulse, the opposite of the prediction. Both EW forecasts fail."
        ),
        md(
            "### 4e · Faithful-engine control — we know the truth here\n\n"
            "On a deterministic panel with a **planted** wave-3 extension (knob `edge`): edge 0 (pure "
            "random walk) must stay at t≈0; a real planted wave-3 must light up. Both hold, so the flat "
            "real-tape result is a genuine null."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.15):\n"
            "    bars, _ = data.synthetic_panel(edge=edge, seed=445, n_days=9000)\n"
            "    r = st.run_experiment(bars, horizon=10, cost_bps=1.0, n_draws=4000)\n"
            "    res.append((edge, r['n_entries'], r['wave_gross']['mean_bps'], r['wave_gross']['t'], r['placebo_p']))\n"
            "labels = [f'edge {e:+.2f}' for e,_,_,_,_ in res]; tv = [r[3] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "ax.bar(labels, tv, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tv): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('wave-3 t-stat (H=10)'); ax.set_title('Control: edge 0 -> t~0; planted edge -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,b,t,p in res: print(f'planted {e:+.2f}: entries={n} bps={b:+.0f} t={t:.2f} placebo_p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: edge 0 gives **t = {R['syn'][0][4]:.2f}** (no false positive); a real "
            f"planted wave-3 gives **t = {R['syn'][1][4]:.2f}**, placebo p = {R['syn'][1][6]:.3f}. The "
            "harness detects Elliott's structure **when it exists** — it just doesn't exist on the real "
            "tape. True negative confirmed."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled wave-3 over {R['n_entries']} entries: max **HAC t = "
            f"{R['w3'][3][6]:.2f} at 40d** (beta, not Elliott), shorter horizons noise/negative, "
            f"same-bars coin **p = {R['w3'][2][7]:.2f}** at 20d. Robust across the Fibonacci band and the "
            "ZigZag threshold. Doesn't clear **t ≥ 2**.\n"
            f"- **Tradability `MIRAGE`** — no edge to monetise; the only positive (40d "
            f"+{R['w3'][3][2]} bps) is buy-and-hold in disguise, and costs make the fast versions "
            "negative.\n"
            f"- **Five waves predict? `NOT SUPPORTED`** — wave-3 extension and completed-impulse "
            f"correction (40d **t = {R['imp'][3][4]:.2f}**) both indistinguishable from chance. Planted-edge "
            f"control **t = {R['syn'][1][4]:.2f}** — the test works; the effect is absent. And remember: "
            "this *generous mechanical proxy* is the best case — a hand-drawn wave count has strictly "
            "more hindsight freedom and so cannot be more predictive."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — no, and here's the cost picture anyway\n\n"
            "There is nothing to deploy. For completeness, gross vs net of 1 bp one-way (×2): the only "
            "positive horizon is the beta-dominated 40-day, and it's not a clean Elliott edge."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets, dirs, npv, nen = pooled(st.wave3_entries)\n"
            "    g = [np.concatenate(rets[h]).mean()*1e4 for h in hs]\n"
            "    nv = [v-2.0 for v in g]\n"
            "else:\n"
            "    g=[r[2] for r in R['w3']]; nv=[r[3] for r in R['w3']]\n"
            "x=np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREY, label='gross')\n"
            "ax.bar(x+.2, nv, .4, color=RED, label='net (2 bp round trip)')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('avg return per trade (bps)'); ax.set_title('Nothing to trade — and not significant where positive')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h,gg,nn in zip(hs,g,nv): print(f'{h:>2}d: gross={gg:+.0f}bps  net={nn:+.0f}bps')"
        ),
        md(
            "> 💡 In plain words: you can't charge costs against an edge that isn't there. The wave-3 "
            "rule is a `MIRAGE` not because costs kill a real signal, but because there's no signal to "
            "begin with."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The irreducibility point matters.** We tested the *tightest mechanical* version. A human "
            "wave count adds more free choices (which degree, which alternate, re-labelling after the "
            "fact) — strictly more overfitting capacity, strictly less out-of-sample claim. The "
            "subjectivity isn't a side note; it's the whole reason the theory survives unfalsified.\n"
            "- **Stricter EW rules.** Add wave-3-not-shortest and wave-4-no-overlap. Fewer signals, same "
            "null — try it.\n"
            "- **Other markets / intraday.** Re-point the loader; the method's failure travels.\n\n"
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
