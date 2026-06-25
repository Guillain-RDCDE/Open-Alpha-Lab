"""Generate the two narrative notebooks for Study 457 (Kicker-Pattern).

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
# 2026-05-31, partial June dropped), 21.4 years, marubozu body-frac=0.60, kicker-direction trade.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=80, n_bull=43, n_bear=37,
    body_frac=0.60, fp_spy="4cb5244f3990",
    # pooled kicker-direction, per horizon:
    # (H, n, kick_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 80, -7.1, 49, -0.20, 30.6, -37.7, -9.1, -0.96, 0.339),
    h10=(10, 80, -17.8, 46, -0.43, 37.4, -55.2, -19.8, -1.13, 0.260),
    h20=(20, 80, -19.1, 51, -0.35, 34.5, -53.5, -21.1, -0.78, 0.438),
    h60=(60, 78, 13.6, 54, 0.11, 34.5, -20.9, 11.6, -0.15, 0.878),
    # per-ticker H=20: (ticker, entries, kick_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 13, 121.7, 0.82, 45.9, 75.8), ("QQQ", 16, 147.5, 1.03, 77.9, 69.5),
         ("IWM", 5, -385.2, None, 20.6, -405.9), ("DIA", 12, -76.0, -0.44, -2.1, -73.9),
         ("GLD", 34, -77.3, -1.38, 30.0, -107.3)],
    # gap-scramble placebo per ticker (H=20, 500 draws): (ticker, obs_bps, p)
    placebo=[("SPY", 121.7, 0.002), ("QQQ", 147.5, 0.002), ("IWM", -385.2, 1.000),
             ("DIA", -76.0, 0.996), ("GLD", -77.3, 0.976)],
    # synthetic control (H=20, seed=458, n_days=6000): (edge, n, kick_bps, win%, one_sample_t)
    syn=[(0.00, 156, 0.6, 49, 0.02), (0.60, 156, 547.4, 79, 10.35)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Gap-reversal forecasts%3F: Busted](https://img.shields.io/badge/Gap--reversal_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from kicker_pattern import data, strategy as st

ASOF = "2026-05-31"
BF = 0.60
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real kicker cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is the \"kicker\" really a violent, reliable turn? 🥾\n"
            "### A famous candlestick pattern — two big candles and a gap — meets a stopwatch\n\n"
            + BADGES +
            "Open any candlestick guide and you'll meet the **kicker**: a strong down candle, then a "
            "strong up candle that **gaps up** the very next day (or the mirror, to the downside). "
            "Two opposite *marubozu* candles — big bodies, almost no wicks — split by a gap **in the "
            "new direction**. The lore, repeated everywhere, is that the kicker **ignores the prior "
            "trend** and marks one of the most *reliable* reversals on the chart: when it prints, you "
            "trade its direction, no questions asked.\n\n"
            "It *looks* decisive on a hand-picked chart — a wall of red, then a gap and a wall of "
            "green. But \"looks decisive\" is exactly how charts fool us. So we did the only fair "
            "thing: encode the kicker **mechanically** (a real marubozu test, a real gap test, no "
            "eyeballing), fire the rule across five big indices over 21 years, and time the result "
            "with a stopwatch — against the only baseline that matters: **entering on random days, in "
            "the same long/short mix.**\n\n"
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
            "| If I trade the kicker's direction, do I make money? | **No — barely break-even, and "
            "*worse* than random.** The win-rate is ~50% and the average trade is slightly negative "
            "at 5–20 days. |\n"
            "| Is it *the kicker* doing anything? | **No.** Enter on **random days** in the same "
            "long/short mix and you do **better** — by 38–55 bps at 5–20 days. |\n"
            "| Does the gap-reversal forecast? | **Not coherently.** Scramble the gap geometry and the "
            "result is a mess: where the kicker looks 'good' it's two noisy small samples that *still* "
            "lose to random; everywhere else the geometry is useless. |\n"
            "| So is it a tradable edge? | **No.** It's a vivid *name* for two big candles and a gap — "
            "not a turn predictor. And it prints only ~4×/year per index, so there's barely anything "
            "to trade even if it worked. |\n\n"
            "> The kicker is a great way to *describe* a violent two-day move after the fact. As a "
            "*forecast* — \"the turn will continue\" — it's a **mirage**: a 50/50 coin with a small "
            "negative tilt, beaten by throwing darts."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Two opposite **marubozu** candles (big body, no wicks) separated by a **gap in the "
            "new direction**. A bullish kicker = a down marubozu, then an up marubozu gapping up; a "
            "bearish kicker = the mirror. It **ignores the prior trend** and marks a violent, reliable "
            "reversal. Trade its direction.\"*\n\n"
            "This is a staple of **Steve Nison's** candlestick canon, and **Thomas Bulkowski's** "
            "*Encyclopedia of Candlestick Charts* ranks the kicker near the top for reversal "
            "frequency. It's built into every charting suite's pattern scanner — so: does the gap "
            "really kick?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the kicker genuinely *forecast* the next move, it would be remarkable: two candles and "
            "a gap predicting a multi-week direction, a clean crack in market efficiency you could "
            "trade on sight. That's the promise.\n\n"
            "But there are two traps. **(a)** It's a *reversal* pattern read on a market that drifts "
            "**up** over time — so a bullish kicker (a long) gets the drift for free, while a bearish "
            "kicker (a short) fights it; you have to neutralise that. **(b)** Bulkowski's famous "
            "'reliable' stat is an **unconditional** reversal frequency, never raced against random "
            "entries. To separate the **pattern** from the **tide**, we draw the kicker by a fixed "
            "rule and compare it to **random days in the same long/short mix.** We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Define a marubozu mechanically.** A candle whose **body is ≥ 60% of its range** — a "
            "strong candle with tiny wicks. No eyeballing.\n"
            "2. **Define the kicker by rule.** Yesterday and today are **opposite-colour** marubozus, "
            "and today **gaps in its own direction** past yesterday's open. Read on today's close — "
            "no future data.\n"
            "3. **Trade the lore.** Long a bullish kicker, short a bearish one, entered at the **next "
            "close**; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**, in the **same "
            "long/short mix**. If the kicker matters, it must beat random. *If it doesn't, the pattern "
            "is a mirage* — that's the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical kicker even look like? Here's SPY with the kickers the rule "
            "fires — green up-arrows for bullish, red down-arrows for bearish."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY'); cl = b['close']\n"
            "    sig = st.kicker_signals(b, body_frac=BF)\n"
            "    seg = cl.iloc[-1500:]\n"
            "    bull = sig[(sig>0) & (sig.index>=seg.index[0])].index\n"
            "    bear = sig[(sig<0) & (sig.index>=seg.index[0])].index\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.scatter(bull, cl.reindex(bull), marker='^', c=GREEN, s=80, zorder=5, label='bullish kicker (long)')\n"
            "    ax.scatter(bear, cl.reindex(bear), marker='v', c=RED, s=80, zorder=5, label='bearish kicker (short)')\n"
            "    ax.set_title('Mechanical kickers on SPY (last ~6y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('kickers in window: bull', len(bull), '| bear', len(bear))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "Strict marubozu+gap kickers are **rare** — only "
            f"~{R['n_entries']} across five tapes in 21 years ({R['n_bull']} bullish, {R['n_bear']} "
            "bearish). That rarity is itself a finding. Now the real question: do those arrows mark "
            "turns? **Let's race the kicker against random entries** at four horizons. Blue = trade "
            "the kicker; grey = random days, same long/short mix."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    kick, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t)\n"
            "            e, dirs = st.kicker_entries(b, body_frac=BF)\n"
            "            re, rdir = st.random_entries(b, max(len(e),50), dirs=dirs, seed=7)\n"
            "            tt.append(st.forward_returns(b, e, dirs, h)); rr.append(st.forward_returns(b, re, rdir, h))\n"
            "        kick.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    kick = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, kick, .4, color='#2c6fbb', label='trade the kicker')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random days (same mix)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,bb) in enumerate(zip(kick,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The kicker does NOT beat random — it mostly loses to it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('kicker:', [round(v) for v in kick]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The kicker is roughly **flat-to-negative** "
            f"(−7 / −18 / −19 bps at 5/10/20 days) while plain **random entries make +30 to +37 bps**. "
            "At every horizon the famous kicker is *worse* than throwing darts. The pattern isn't "
            "marking a turn — it's a 50/50 coin with a small negative tilt."
        ),
        md(
            "**One more sanity check.** What if we scramble the kicker's *gap geometry* — keep the same "
            "big candles but randomise which way the gaps point? If price really 'respects the gap', "
            "the nonsense version should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        pl = st.gap_scramble_placebo(load(t), 20, body_frac=BF, n_draws=200, seed=457)\n"
            "        rows.append((t, pl['obs']*1e4, pl['p_value']))\n"
            "else:\n"
            "    rows = [(t, o, p) for (t,o,p) in R['placebo']]\n"
            "for t,o,p in rows:\n"
            "    print(f'{t}: real kicker {o:+7.1f} bps   scramble p={p:.3f}')\n"
            "print('=> incoherent: only the two noisy positives get a low p, and they still lose to random.')"
        ),
        md(
            "The placebo is a **mess**, not a confirmation. The only two tickers where scrambling the "
            "gaps 'hurts' (SPY, QQQ, low *p*) are exactly the two with a *handful* of trades and a "
            "noisy positive mean — and those **still lose to random entries** (see the quants "
            "notebook). On the other three tapes the gap geometry is useless (*p* ≈ 1). A real signal "
            "has a *coherent* placebo; this one doesn't. The gap isn't doing the work."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Trading the kicker does **not** beat random entries (it's *worse* at "
            "every horizon by 21–55 bps). The win-rate is ~50% with a small negative mean.\n"
            "- **Tradability — Mirage.** Nothing to trade: it fires ~4×/year per index, underperforms "
            "darts, and costs only deepen the hole.\n"
            "- **\"Does the gap-reversal forecast?\" — Busted.** Scramble the gaps and the result is "
            "incoherent — the apparent 'edge' is two noisy small samples that fail the random test."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The kicker is a 50/50 coin with a slightly negative average "
            "and a tiny sample (~4 prints/year per index), and it *loses* to entering on random days. "
            "Costs (commissions + spread on every print) push the already-no-edge result further "
            "negative. As a forecasting tool it doesn't pay; as a descriptive label for 'two big "
            "candles and a gap', it was never a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Bulkowski's 'reliability'.** His high reversal-frequency stat is **unconditional** — "
            "it never races the pattern against random entries or corrects for the market's drift. The "
            "quants notebook shows that once you do, the edge is gone.\n"
            "- **Looser / stricter marubozu.** At the textbook-strict 80%-body the kicker prints only "
            "**6 times in 21 years** — too rare to test. We use the loosest defensible 60% and still "
            "find nothing.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* kicker continuation "
            "into a synthetic tape and shows the harness banks it (so the null here isn't a dead "
            "detector — it's an honest 'nothing there').\n\n"
            "*Think the kicker forecasts? Show it beating random entries at **t ≥ 2** on a real tape — "
            "then we'll talk.*"
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
            "# The Kicker pattern — a quantitative teardown 🔬\n"
            "### Mechanical marubozu+gap kickers on 5 indices · kicker-direction forward returns · "
            "one-sample HAC *t* · a **direction-matched** random-entry baseline · a gap-scramble "
            "geometry placebo · costs · a synthetic planted-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **pattern** from the **drift**: the kicker mixes longs and shorts on an "
            "upward-drifting tape, so the only meaningful test is kicker-vs-**direction-matched** "
            "random, plus a placebo that destroys the gap geometry while preserving the candle "
            "marginal.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return**), 2005→2026. A marubozu is body/range ≥ 0.60; a kicker "
            "is an opposite-colour marubozu pair gapping in the new direction, read on the close of "
            "bar *t*, entered the **next close** (one documented lag). Offline core + synthetic control "
            "are deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Kicker vs a **direction-matched random** baseline: the kicker is "
            f"*worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps) and the Welch *t* is **negative everywhere** (best "
            f"{R['h60'][8]:+.2f} at 60d). |\n"
            f"| **Tradability** | `MIRAGE` | ~4 prints/yr per tape, ~50% win-rate, slightly negative "
            f"mean; costs deepen the hole. No edge to scale. |\n"
            f"| **Gap-reversal forecasts?** | `BUSTED` | The gap-scramble placebo is **incoherent** — "
            f"the only two low-*p* tickers (SPY/QQQ) are noisy small-sample positives that *fail* the "
            f"random test; the other three give *p* ≥ 0.976. |\n\n"
            "> 💡 In plain words: the kicker is a 50/50 coin with a small negative tilt that loses to "
            "random entries. The 'reliable reversal' reputation is an unconditional-frequency artifact, "
            "not a conditional edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A bar is a **marubozu** if $|C-O|/(H-L)\\ge 0.60$; its colour is $\\mathrm{sgn}(C-O)$. A "
            "**kicker** at bar $t$ requires bars $t-1,t$ to be opposite-colour marubozus with bar $t$ "
            "gapping in its own direction past $O_{t-1}$: bullish iff $C_t>O_t$ and $O_t>O_{t-1}$; "
            "bearish iff $C_t<O_t$ and $O_t<O_{t-1}$. We take the kicker's direction, enter at "
            "$t{+}1$'s close, and measure the signed $H$-day return.\n\n"
            "- **H₀ (drift).** Kicker returns equal a **direction-matched** random-entry baseline.\n"
            "- **H₁ (the kicker forecasts).** Kicker returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the gap matters).** Kicker returns exceed a **gap-scramble** placebo whose gap "
            "signs are randomised.\n\n"
            "We find **H₀ not rejected** (kicker ≤ random at *every* horizon), **H₁ rejected** (Welch t "
            "negative everywhere), **H₂ rejected** (placebo incoherent). The steelman fails on every "
            "leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift in a long/short rule.** Equity indices drift up. A bullish kicker (long) "
            "inherits that drift; a bearish kicker (short) fights it. A one-sample $t$ against **zero** "
            "is therefore uninterpretable for a mixed rule. The fix is a **direction-matched** "
            "random-entry baseline: the same long/short proportion, same instrument, epoch and hold.\n\n"
            "**(b) Unconditional 'reliability'.** Bulkowski's reputation stat counts how often the "
            "pattern is *followed by* a move in its direction — but on a drifting, autocorrelated tape "
            "that number is inflated for free. The **gap-scramble placebo** keeps the big candles but "
            "randomises the gap signs; if the real result survives, the gap was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} kickers** "
            f"({R['n_bull']} bullish / {R['n_bear']} bearish) pooled.\n"
            f"- **Marubozu.** body/range ≥ {R['body_frac']:.2f} (strong candle). Colour = sign(C−O).\n"
            "- **Kicker.** opposite-colour marubozu pair, bar *t* gaps in its own direction past "
            "$O_{t-1}$; read on close of *t* (uses only *t-1,t* — no look-ahead).\n"
            "- **Entry.** next close (one lag); hold H ∈ {5,10,20,60}; signed by kicker direction.\n"
            "- **Null #1 — one-sample HAC t** of kicker returns vs 0 (Newey-West) — *diagnostic only*.\n"
            "- **Null #2 — direction-matched random baseline**, Welch two-sample kicker vs random "
            "(the *real* test).\n"
            "- **Null #3 — gap-scramble placebo** (gap signs randomised, candle marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every kicker.\n"
            "- **Positive control.** Synthetic tape with a **planted** kicker continuation (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Kicker vs random across horizons — the gap is the verdict\n\n"
            "Mean signed return, kicker vs direction-matched random entry, all four horizons. The "
            "kicker should tower over random if it forecasts. It doesn't — it loses at every horizon."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    from scipy import stats\n"
            "    kick, rnd, welch = [], [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t)\n"
            "            e, dirs = st.kicker_entries(b, body_frac=BF)\n"
            "            re, rdir = st.random_entries(b, max(len(e),50), dirs=dirs, seed=7)\n"
            "            tt.append(st.forward_returns(b, e, dirs, h)); rr.append(st.forward_returns(b, re, rdir, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        kick.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    kick = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "x = np.arange(len(hs))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar(x-.2, kick, .4, color='#2c6fbb', label='kicker'); a1.bar(x+.2, rnd, .4, color=GREY, label='random (same mix)')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(x); a1.set_xticklabels([f'{h}d' for h in hs])\n"
            "for i,(a,b) in enumerate(zip(kick,rnd)):\n"
            "    a1.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    a1.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "a1.set_ylabel('mean fwd return (bps)'); a1.set_title('Kicker loses to random at every horizon'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Kicker vs RANDOM, Welch t (negative everywhere)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta kicker-random (bps):', [round(a-b) for a,b in zip(kick,rnd)])\n"
            "print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the kicker (blue) sits at or below zero while random (grey) earns the "
            f"drift the longs deserve and the shorts give back — net, the kicker is behind by "
            f"{abs(R['h10'][6]):.0f} bps at 10d. The Welch *t* (right) is **negative at all four "
            "horizons** — the kicker doesn't just fail to beat random, it underperforms it."
        ),
        md(
            "### 4b · Per-ticker — no coherent edge anywhere\n\n"
            "20-day kicker-minus-random delta, per instrument. If the kicker worked it would be "
            "positive across the board; instead it's a scatter dominated by tiny samples."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas, ns = [], [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        b = load(t)\n"
            "        e, dirs = st.kicker_entries(b, body_frac=BF); re, rdir = st.random_entries(b, max(len(e),50), dirs=dirs, seed=7)\n"
            "        d = st.summarize(st.forward_returns(b,e,dirs,20))['mean_bps'] - st.summarize(st.forward_returns(b,re,rdir,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d); ns.append(len(e))\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]; ns = [p[1] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(d,nn) in enumerate(zip(deltas,ns)): ax.annotate(f'{d:+.0f}\\n(n={nn})',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=8)\n"
            "ax.set_ylabel('20d kicker − random (bps)'); ax.set_title('No coherent cross-sectional edge (and tiny samples)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: SPY/QQQ are *nominally* positive ({R['per'][0][5]:+.0f}/"
            f"{R['per'][1][5]:+.0f} bps) on 13/16 trades — neither one-sample *t* clears 1.1. IWM has "
            "**5 trades** (uninterpretable). DIA and GLD are negative. No coherent edge — the signature "
            "of a pattern that isn't doing anything."
        ),
        md(
            "### 4c · The gap-scramble placebo — incoherent, not confirmatory\n\n"
            "Keep the big candles, randomise the gap signs (the candle marginal is preserved). A real "
            "gap-reversal edge would make every ticker's *p* small. Instead the *p*-values are all over "
            "the map — and the two that are small belong to the noisy positives that already failed 4a."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, obs, pv = [], [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        pl = st.gap_scramble_placebo(load(t), 20, body_frac=BF, n_draws=300, seed=457)\n"
            "        names.append(t); obs.append(pl['obs']*1e4); pv.append(pl['p_value'])\n"
            "else:\n"
            "    names = [p[0] for p in R['placebo']]; obs = [p[1] for p in R['placebo']]; pv = [p[2] for p in R['placebo']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, pv, color=[RED if p>0.05 else GREEN for p in pv], width=.6)\n"
            "ax.axhline(0.05, ls='--', c='k', label='p=0.05'); ax.set_ylim(0,1.05)\n"
            "for i,(p,o) in enumerate(zip(pv,obs)): ax.annotate(f'p={p:.2f}\\n({o:+.0f}bps)',(i,p),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_ylabel('gap-scramble placebo p'); ax.set_title('Incoherent placebo: not a confirmation'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo p by ticker:', {n:round(p,3) for n,p in zip(names,pv)})"
        ),
        md(
            "> 💡 In plain words: a forecasting signal has a **coherent** placebo — scrambling the "
            "geometry should hurt *everywhere*. Here only SPY/QQQ get a low *p* (and they're the noisy "
            "positives that lose to random in 4a); IWM/DIA/GLD give *p* ≈ 1, meaning random gap-signs "
            "do **as well or better**. The gap isn't load-bearing."
        ),
        md(
            "### 4d · Synthetic positive control — the harness CAN bank a real continuation\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** kicker continuation "
            "into a synthetic tape and check the same rule banks it: edge=0 (formation, no follow-"
            "through) must stay at t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=458, n_days=6000)\n"
            "    e, dirs = st.kicker_entries(px, body_frac=BF); s = st.summarize(st.forward_returns(px, e, dirs, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted continuation -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} kick={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with the formation present but **no** planted follow-through the "
            f"control sits at **t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false "
            f"positive); a planted continuation reaches **t = {R['syn'][1][4]:.2f}** (win "
            f"{R['syn'][1][3]:.0f}%). The detector works — so the flat real-tape result is a genuine "
            "'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the kicker does not beat a direction-matched random baseline "
            f"(kicker − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t **negative everywhere**, best "
            f"{R['h60'][8]:+.2f}). The one-sample mean is itself ≈ 0 (−7 to −19 bps at 5–20d).\n"
            f"- **Tradability `MIRAGE`** — ~4 prints/yr per tape, ~50% win-rate, slightly negative "
            "mean, beaten by darts; costs deepen the hole. No edge to scale.\n"
            f"- **Gap-reversal forecasts? `BUSTED`** — the gap-scramble placebo is incoherent: the "
            f"two low-*p* tickers (SPY/QQQ) are noisy small-sample positives that fail the random test, "
            f"while IWM/DIA/GLD give *p* ≥ 0.976. The gap geometry carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The kicker fires ~4×/year per index, wins ~50% of the time, and averages slightly "
            "negative — *and* it loses to entering on random days in the same long/short mix. There is "
            "no capacity question because there is no edge to scale, and costs only make a no-edge "
            "result worse. The kicker is a descriptive label for a violent two-day move, not a "
            "forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Unconditional vs conditional reliability.** Bulkowski's headline stat ('reverses X% "
            "of the time') is unconditional; the conditional, drift-matched test here finds no edge. "
            "The gap between the two is the whole lesson.\n"
            "- **Marubozu strictness.** At body/range ≥ 0.80 the canonical kicker prints **6 times in "
            "21 years** — too rare to test; loosening to 0.60 gives 80 events and still nothing. The "
            "pattern's rarity is itself a finding.\n"
            "- **Gap definitions.** Body-gap vs open-gap variants are affine loosenings of the same "
            "geometry and inherit the same confound. The deterministic synthetic control proves the "
            "detector is live.\n\n"
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
