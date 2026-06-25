"""Generate the two narrative notebooks for Study 453 (Three-Inside-Up / Down).

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
# 2026-05-31, partial June dropped), 21.4 years, trend lookback=5, confirmed three-inside-up long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=64, tl=5,
    fp_spy="4cb5244f3990",
    # pooled confirmed three-inside-up, per horizon:
    # (H, n, conf_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 64, 30.4, 61, 1.22, 10.3, 20.1, 28.4, 0.63, 0.528),
    h10=(10, 64, 10.3, 62, 0.25, 53.6, -43.2, 8.3, -0.86, 0.392),
    h20=(20, 64, -106.9, 47, -1.61, 33.1, -139.9, -108.9, -1.75, 0.083),
    h60=(60, 64, -13.3, 48, -0.12, 118.9, -132.3, -15.3, -1.04, 0.302),
    # per-ticker H=20: (ticker, entries, conf_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 11, -177.4, -1.08, 39.3, -216.7), ("QQQ", 14, -136.3, -0.62, 88.2, -224.5),
         ("IWM", 18, -107.6, -1.21, 2.7, -110.3), ("DIA", 11, -93.3, -0.97, 33.7, -127.0),
         ("GLD", 10, -1.6, -0.01, 1.5, -3.1)],
    # confirmation-candle thesis test (harami-only placebo), pooled H=20:
    # (n_conf, conf_bps, n_har, har_bps, delta_bps, welch_t, welch_p)
    placebo=(64, -106.9, 211, 91.4, -198.3, -2.49, 0.015),
    # synthetic control (H=20, n_days=8000): (edge, n, conf_bps, win%, one_sample_t)
    syn=[(0.00, 20, -87.1, 45, -1.18), (0.60, 20, 341.6, 75, 4.43)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Confirmation_adds_edge%3F: Busted](https://img.shields.io/badge/Confirmation_adds_edge%3F-Busted-8b949e?style=flat-square)\n\n"
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

from three_inside import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real three-inside cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the \"three-inside-up\" candle pattern flip the trend? 🕯️\n"
            "### A textbook three-candle reversal — down bar, inside harami, a confirming candle — "
            "meets a stopwatch\n\n"
            + BADGES +
            "Open any candlestick book and you'll meet the **three-inside-up**: a down candle, then a "
            "small **inside** candle tucked inside it (a *harami*), then a third **confirmation** "
            "candle that closes back up past the first. The lore — from Nison and every chart-pattern "
            "site — is that this triplet **flips the trend**: the confirming candle says \"the "
            "down-move is over, buy.\" The whole appeal is that *third* candle: it's supposed to be "
            "the proof.\n\n"
            "It *looks* convincing on a hand-picked chart. But a pattern you spot **after** the bars "
            "have printed is the textbook way to fool yourself. So we did the only fair thing: encode "
            "the pattern **mechanically** (no eyeballing), fire it across five big indices over 21 "
            "years, and time the result — against the baseline that matters (**buying on random "
            "days**) and against the **same pattern with the confirming candle removed**.\n\n"
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
            "| If I buy when a three-inside-up confirms, do I make money? | **No, on average you "
            "*lose*.** Over 20 days the pattern averages **−107 bps** — the \"bullish reversal\" is "
            "typically followed by a *drop*. |\n"
            "| Does it beat buying on **random days**? | **No.** Random-day entries make **+33 bps** "
            "over the same 20 days; the pattern is **140 bps worse** than throwing darts. |\n"
            "| Does the famous **confirmation candle** add anything? | **It makes it worse.** Drop the "
            "third candle and just buy the bare harami → **+91 bps**. The confirmation candle "
            "*subtracts* ~198 bps. |\n"
            "| So is it a tradable edge? | **No.** It's a rare pattern (64 signals in 21 years) that "
            "loses to a coin flip and whose celebrated \"proof\" candle is the worst part. |\n\n"
            "> The three-inside-up is a tidy way to *label* a wiggle after the fact. As a *forecast* — "
            "\"the trend will flip up\" — it's a **mirage**, and the confirming candle (the bit "
            "everyone trusts) actively hurts."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"After a downtrend: a down candle (A), then a small candle that fits **inside** A's "
            "body (B, the harami), then a third candle (C) that closes back **above A's open** — that "
            "confirms the reversal. Go long. The mirror three-inside-down is the sell at a top.\"*\n\n"
            "This is a **confirmed harami**, popularised by **Steve Nison** and catalogued by "
            "**Gregory Morris** — one of the most-taught three-candle reversals in technical analysis, "
            "built into every charting suite's pattern scanner. The distinguishing feature versus the "
            "bare two-candle harami is the **confirmation candle**. So the real question is: does that "
            "third candle earn its keep?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the three-inside-up genuinely *forecast* reversals, it would be remarkable: three past "
            "candles predicting a turning point, a clean edge you could scan for. That's the dream the "
            "pattern sells.\n\n"
            "But there are two traps. First, it's spotted **after** the candles print on a market that "
            "drifts **up**, so *any* long rule looks okay by default — we must compare to buying on "
            "**random days**. Second, the pattern's selling point is the **confirmation candle**, and "
            "the honest way to test it is to **remove it**: fire on the bare harami and see whether the "
            "third candle added or destroyed edge. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Detect the pattern by rule.** A is a down candle after a 5-day downtrend; B's whole "
            "range sits inside A's body (a strict harami); C closes back above A's open and above B's "
            "close. All read on **closed** bars — no peeking ahead.\n"
            "2. **Trade the lore.** On the confirming candle's close, buy at the **next** close; "
            "measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the same hold on **random days**. If the pattern matters, "
            "it must beat random.\n"
            "4. **The confirmation test.** Re-run with the **confirmation candle removed** (buy the "
            "bare harami). If the third candle adds edge, the confirmed version must win. *If it "
            "doesn't, the celebrated confirmation is busted* — announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical three-inside-up even look like? Here's SPY with the confirmed "
            "buy signals the rule fires."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-500:]\n"
            "    ent = st.three_inside_entries(b, trend_lookback=R['tl'], require_confirm=True)\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg['close'].values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.scatter(ent, b['close'].reindex(ent), c=GREEN, s=55, zorder=5, marker='^', label='three-inside-up BUY')\n"
            "    ax.set_title('Mechanical three-inside-up signals on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('confirmed signals in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The signals are rare and land on small bounces. The question is whether those green "
            "arrows are followed by sustained up-moves. **Let's race the confirmed pattern against "
            "random entries** at four horizons. Blue = the three-inside-up; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    conf, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.three_inside_entries(bb, trend_lookback=R['tl'], require_confirm=True)\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        conf.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    conf = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, conf, .4, color='#2c6fbb', label='three-inside-up')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,bb) in enumerate(zip(conf,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom' if bb>=0 else 'top',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The pattern does NOT beat random — and goes negative at 20d'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('confirmed:', [round(v) for v in conf]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story. The \"bullish reversal\" makes money only at 5 days "
            f"(**+{R['h5'][2]:.0f} bps**), then turns **negative at 20 days** (**{R['h20'][2]:.0f} "
            f"bps**) — the down-move is, on average, *not* over. And random entries do better at "
            "10/20/60 days. The pattern doesn't beat a dart."
        ),
        md(
            "**The key test.** What if we **remove the confirmation candle** — buy the bare harami "
            "instead? If the famous third candle adds edge, the confirmed version should win."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cf, hr = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        ec = st.three_inside_entries(bb, trend_lookback=R['tl'], require_confirm=True)\n"
            "        eh = st.three_inside_entries(bb, trend_lookback=R['tl'], require_confirm=False)\n"
            "        cf.append(st.forward_returns(c, ec, 20)); hr.append(st.forward_returns(c, eh, 20))\n"
            "    conf_m = np.concatenate(cf).mean()*1e4; har_m = np.concatenate(hr).mean()*1e4\n"
            "else:\n"
            "    conf_m = R['placebo'][1]; har_m = R['placebo'][3]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['bare harami\\n(no confirm)','three-inside-up\\n(confirmed)'], [har_m, conf_m],\n"
            "       color=[GREEN, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate([har_m, conf_m]): ax.annotate(f'{v:+.0f} bps',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('mean 20d return (bps)'); ax.set_title('The confirmation candle makes it WORSE')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'bare harami: {har_m:+.0f} bps   confirmed: {conf_m:+.0f} bps   confirmation adds {conf_m-har_m:+.0f} bps')"
        ),
        md(
            f"There it is. The bare harami averages **+{R['placebo'][3]:.0f} bps**; adding the "
            f"\"confirming\" candle drags it to **{R['placebo'][1]:.0f} bps** — a "
            f"**{R['placebo'][4]:.0f} bps** swing the *wrong* way. The confirmation candle, the part "
            "the lore prizes most, is the worst part of the rule: it makes you buy *after* a one-day "
            "pop, i.e. chase."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The confirmed three-inside-up does **not** beat buying on random "
            "days (it's worse at 10/20/60 days and goes negative at 20). There isn't even a drift "
            "mirage here — the \"reversal\" is, on average, followed by a fall.\n"
            "- **Tradability — Mirage.** A rare pattern (64 signals in 21 years) that loses to a coin "
            "flip in every name; costs only deepen it.\n"
            "- **\"Does the confirmation candle add edge?\" — Busted.** Removing the third candle "
            "*improves* the result by ~198 bps. The celebrated confirmation is a net negative."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The pattern fires rarely, loses to a random-day entry, and "
            "its defining feature (the confirmation candle) actively hurts. If anything, the *bare* "
            "harami is the better idea — but even that is just a slow dip-buy that you'd capture more "
            "cheaply by holding the index. Costs on every signal push the already-negative result "
            "further down. As a forecasting tool, the three-inside-up doesn't pay."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The bare harami.** It beat the confirmed version here — a fun follow-up is to test "
            "the two-candle harami on its own terms (it's still no edge vs random, but it's *less* "
            "bad).\n"
            "- **The mirror three-inside-down.** The bearish version at tops inherits the same "
            "problem in reverse — confirmation = chasing.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* three-inside-up "
            "bounce into a synthetic tape and shows the harness banks it (so the flat real-tape result "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the confirmation candle forecasts? Show the confirmed three-inside-up beating both "
            "random entries **and** the bare harami at **t ≥ 2** on a real tape — then we'll talk.*"
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
            "# Three-Inside-Up / Down — a quantitative teardown 🔬\n"
            "### Mechanical harami + confirmation on 5 indices · forward returns · one-sample HAC *t* "
            "· a drift-matched random-entry baseline · a confirmation-candle (harami-only) placebo · "
            "costs · a synthetic planted-reversal control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is twofold: (1) separate the **pattern** from the **drift** (random-entry baseline), and "
            "(2) isolate the **confirmation candle's** marginal contribution by re-running on the bare "
            "harami. The thesis on trial is exactly that second question.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. The pattern is read on "
            "**closed** bars; entry is the **next close** (one documented lag). Offline core + "
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
            f"| **Signal** | `NONE` | Confirmed three-inside-up vs a **drift-matched random** "
            f"baseline: Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; the Welch *t* **never clears 2** (min {R['h20'][8]:+.2f} at 20d). "
            "The rule's own one-sample *t* never clears 2 and turns negative at 20d. |\n"
            f"| **Tradability** | `MIRAGE` | Only **{R['n_entries']}** signals in 21 years across 5 "
            f"tapes; the confirmed entry loses to random in **5 of 5** names (20d t = {R['h20'][4]:.2f}, "
            "negative). No drift to inherit, nothing to scale. |\n"
            f"| **Confirmation adds edge?** | `BUSTED` | Dropping the confirmation candle (harami-only "
            f"placebo) **improves** the 20d return by **{abs(R['placebo'][4]):.0f} bps** "
            f"({R['placebo'][3]:+.0f} vs {R['placebo'][1]:+.0f}); Welch *t* = {R['placebo'][5]:+.2f} "
            f"(*p* = {R['placebo'][6]:.3f}). The third candle is a *negative* contributor. |\n\n"
            "> 💡 In plain words: the pattern doesn't even ride the drift (it loses to random), and "
            "the bit everyone trusts — the confirmation candle — is the worst part of the rule."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "After a downtrend, bars A,B,C with bodies $[\\,\\underline b_i,\\overline b_i\\,]$. "
            "**A** is a down candle ($C_A<O_A$); **B** is a harami "
            "($\\overline h_B\\le\\overline b_A$ and $\\underline\\ell_B\\ge\\underline b_A$, the "
            "range inside A's body); **C** confirms ($C_C>O_A$ and $C_C>C_B$). The rule buys on C's "
            "close (entered next close) and holds H days.\n\n"
            "- **H₀ (drift).** Confirmed returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the pattern forecasts).** Confirmed returns **exceed** random at some horizon, "
            "t ≥ 2.\n"
            "- **H₂ (the confirmation candle matters).** Confirmed returns **exceed** the bare "
            "**harami-only** entry (confirmation removed).\n\n"
            "We find **H₀ not rejected** (confirmed ≤ random at 10–60d), **H₁ rejected** (Welch t "
            "never ≥ 2; the rule even goes negative), and **H₂ rejected the wrong way** (confirmation "
            "*subtracts* edge, Welch t ≈ −2.5). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean; a one-sample $t$ "
            "of a long-only rule against **zero** measures the tide, not the tool. The fix is the "
            "**random-entry baseline** (same instrument, epoch, hold) and a Welch test of "
            "confirmed-*minus*-random. (Here the rule loses even this, which is a stronger refutation.)"
            "\n\n**(b) The confirmation candle as the load-bearing claim.** The three-inside-up's whole "
            "selling point over the two-candle harami is the confirming third candle. The honest test "
            "is to **remove it** — the **harami-only placebo** keeps the down-candle-plus-inside-bar "
            "event but drops the confirmation, so the difference is the confirmation candle's *marginal* "
            "contribution. If it's ≤ 0, the third candle is decorative (or worse)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} confirmed signals** "
            "pooled.\n"
            f"- **Pattern.** A: down candle after a {R['tl']}-day downtrend. B: range inside A's body "
            "(strict harami). C: close > A's open and > B's close. All on **closed** bars.\n"
            "- **Entry.** On C's close; enter **next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of confirmed returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample confirmed vs random (the drift "
            "test).\n"
            "- **Null #3 — harami-only placebo** (confirmation removed), Welch confirmed vs harami "
            "(**the thesis test**).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every signal.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-three-inside-up bounce "
            "(knob `edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · One-sample t vs the random baseline\n\n"
            "Left: the confirmed pattern's **one-sample** t against zero. Right: the same confirmed "
            "entry vs a **drift-matched random** baseline (Welch). Neither clears 2; both go negative "
            "at 20d."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, conf, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.three_inside_entries(bb, trend_lookback=R['tl'], require_confirm=True)\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); conf.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    conf = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar'); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Confirmed vs RANDOM, Welch t (never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: there's no beta mirage to strip here — the confirmed pattern's own "
            f"one-sample t never clears 2 (20d **{R['h20'][4]:.2f}**, negative). Against random it's "
            f"negative at 10/20/60d (Welch {R['h20'][8]:+.2f} at 20d). The pattern doesn't forecast a "
            "reversal; it mostly precedes more weakness."
        ),
        md(
            "### 4b · Confirmed vs random across horizons\n\n"
            "Mean return, confirmed three-inside-up vs random entry, all four horizons. The pattern "
            "should tower over random if it forecasts. It doesn't — and goes negative at 20d."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, conf, .4, color='#2c6fbb', label='three-inside-up (confirmed)')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,b) in enumerate(zip(conf,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom' if b>=0 else 'top',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Confirmed three-inside-up does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta confirmed-random (bps):', [round(a-b) for a,b in zip(conf,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the confirmed pattern is **{R['h20'][2]:.0f} bps** while "
            f"random is **+{R['h20'][5]:.0f} bps** — the famous reversal *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. Only at 5d is it ahead, and the Welch test (4a) says even "
            "that is noise."
        ),
        md(
            "### 4c · The confirmation-candle placebo — the thesis test\n\n"
            "Drop the confirmation candle and fire on the bare harami (same down-candle-plus-inside-bar "
            "event, no third candle). If the confirmation adds edge, the confirmed bar should beat the "
            "harami bar. It's the **opposite**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    from scipy import stats\n"
            "    cf, hr = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        ec = st.three_inside_entries(bb, trend_lookback=R['tl'], require_confirm=True)\n"
            "        eh = st.three_inside_entries(bb, trend_lookback=R['tl'], require_confirm=False)\n"
            "        cf.append(st.forward_returns(c, ec, 20)); hr.append(st.forward_returns(c, eh, 20))\n"
            "    cf = np.concatenate(cf); hr = np.concatenate(hr)\n"
            "    conf_m = cf.mean()*1e4; har_m = hr.mean()*1e4\n"
            "    wt, wp = stats.ttest_ind(cf, hr, equal_var=False)\n"
            "    n_c, n_h = cf.size, hr.size\n"
            "else:\n"
            "    n_c, conf_m, n_h, har_m, _, wt, wp = R['placebo']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.2))\n"
            "ax.bar([f'bare harami\\n(n={n_h})','three-inside-up\\n(confirmed, n={})'.format(n_c)],\n"
            "       [har_m, conf_m], color=[GREEN, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate([har_m, conf_m]): ax.annotate(f'{v:+.0f} bps',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_ylabel('mean 20d return (bps)'); ax.set_title(f'Confirmation candle adds {conf_m-har_m:+.0f} bps (Welch t={wt:+.2f}, p={wp:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'harami-only: {har_m:+.1f} bps (n={n_h})   confirmed: {conf_m:+.1f} bps (n={n_c})')\n"
            "print(f'confirmation marginal = {conf_m-har_m:+.1f} bps   Welch t={wt:+.2f}  p={wp:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the bare harami averages **{R['placebo'][3]:+.0f} bps**; the "
            f"confirmed version **{R['placebo'][1]:+.0f} bps**. The confirmation candle's marginal "
            f"contribution is **{R['placebo'][4]:+.0f} bps** — significantly *negative* (Welch "
            f"t = {R['placebo'][5]:+.2f}, p = {R['placebo'][6]:.3f}). Requiring an up-close third "
            "candle means buying after a one-day pop: you chase, and pay for it. **Thesis busted.**"
        ),
        md(
            "### 4d · Per-ticker (H = 20) — confirmed loses to random everywhere\n\n"
            "20-day confirmed-minus-random delta, per instrument. If the pattern worked it would be "
            "positive; it's negative in all 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.three_inside_entries(bb, trend_lookback=R['tl'], require_confirm=True)\n"
            "        re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d confirmed − random (bps)'); ax.set_title('Confirmed underperforms random in 5 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: every name is negative, from GLD's tiny **{R['per'][4][5]:+.0f} bps** "
            f"to QQQ's **{R['per'][1][5]:+.0f} bps**. No coherent cross-sectional edge — the pattern "
            "is rare and, where it fires, it loses to a dart."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real reversal\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-three-inside-up "
            "bounce into a synthetic tape and check the same confirmed rule banks it: edge=0 must stay "
            "below significance; edge>0 must light up."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=453, n_days=8000)\n"
            "    c = px['close']; e = st.three_inside_entries(px, trend_lookback=R['tl'], require_confirm=True)\n"
            "    s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom' if t>=0 else 'top')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t<0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} conf={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"bounce reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat/negative real-tape result is a genuine 'nothing there', not a broken "
            "pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the confirmed three-inside-up does not beat a drift-matched random "
            f"baseline (confirmed − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t never clears 2). The rule's own "
            f"one-sample t never clears 2 and turns negative (20d **{R['h20'][4]:.2f}**) — no beta "
            "mirage, just no edge.\n"
            f"- **Tradability `MIRAGE`** — {R['n_entries']} signals in 21 years, losing to random in 5 "
            "of 5 names; costs only deepen the hole. Nothing to scale.\n"
            f"- **Confirmation adds edge? `BUSTED`** — the harami-only placebo *improves* the 20d "
            f"return by **{abs(R['placebo'][4]):.0f} bps** ({R['placebo'][3]:+.0f} vs "
            f"{R['placebo'][1]:+.0f}; Welch t = {R['placebo'][5]:+.2f}, p = {R['placebo'][6]:.3f}). "
            "The confirmation candle — the heart of the pattern — is a *negative* contributor."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The confirmed three-inside-up fires rarely, loses to a random-day entry in every name, "
            "and its defining confirmation candle subtracts edge. There is no drift to inherit and no "
            "residual signal to scale; costs on each rare signal make it worse. The pattern is a "
            "descriptive label for a wiggle, not a forecasting strategy — and the part of it the lore "
            "trusts most is the part that hurts."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The bare harami on its own terms.** It beat the confirmed version here; a follow-up "
            "tests whether the two-candle harami clears the random baseline (spoiler: still no, but "
            "it's *less* bad — the confirmation is the active poison).\n"
            "- **The mirror three-inside-down.** The bearish top pattern inherits the same "
            "confirmation-as-chasing problem in reverse.\n"
            "- **Confirmation strength.** Vary the confirmation threshold (close past A's high vs A's "
            "open); the stronger the required up-close, the more you chase — a clean monotone of the "
            "negative-contribution result.\n\n"
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
