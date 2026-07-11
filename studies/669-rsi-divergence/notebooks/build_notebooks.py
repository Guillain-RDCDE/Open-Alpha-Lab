"""Generate the two narrative notebooks for Study 669 (RSI-Divergence).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached six-ticker OHLC
tape under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY/QQQ/IWM/AAPL/
# MSFT/NVDA 2010-01-04 -> 2026-06-30; 109 algorithmically confirmed bullish divergences).
R = dict(
    start="2010-01-04", end="2026-06-30", n_events=109,
    by_ticker={"AAPL": 16, "IWM": 16, "MSFT": 22, "NVDA": 22, "QQQ": 15, "SPY": 18},
    n_base=24743,
    # headline: h -> (sig_bps, base_bps, gap_bps, welch_t, nw_t, n_sig, hit_pct, wilson_lo, wilson_hi)
    headline={
        5:  (+1.89, +31.60, -29.71, -0.62, -0.63, 109, 52.3, 43.0, 61.4),
        10: (+57.58, +68.83, -11.25, -0.20, -0.20, 109, 56.0, 46.6, 64.9),
        20: (+84.08, +145.26, -61.17, -0.84, -0.82, 109, 56.9, 47.5, 65.8),
    },
    # placebo: h -> (obs_bps, mean_bps, sd_bps, p_value)
    placebo={
        5:  (+1.89, +32.19, 36.12, 0.799),
        10: (+57.58, +71.01, 50.38, 0.613),
        20: (+84.08, +150.53, 71.43, 0.827),
    },
    era_split="2019-10-01", era_early=+20.16, era_early_n=68,
    era_late=+119.64, era_late_n=41, era_diff_t=+0.83,
    tw_gross=+57.58, tw_net5=+47.58, tw_net10=+37.58, tw_n=109,
    tw_hit=56.0, tw_hit_lo=46.6, tw_hit_hi=64.9,
    tw_worst=-1434.8, tw_best=+1809.2,
    uncond_hit10=59.7, uncond_n10=24822,
    syn_null_mean=+0.31, syn_null_sd=1.08, syn_null_fire=0,
    syn_planted_t=+6.96, syn_planted_nw=+7.09, syn_planted_n=102, syn_planted_mean=+179.3,
    fp={"SPY": "249b0ef2b1da", "QQQ": "dd77b24463c4", "IWM": "3d13a729a6a7",
        "AAPL": "1d9e6ffa4dab", "MSFT": "db3a2e6ab5ea", "NVDA": "4f4ce66b8f52"},
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_random_signal%3F: Busted](https://img.shields.io/badge/Beats_a_random_signal%3F-Busted-8b949e?style=flat-square)\n\n"
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

from rsi_divergence import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    TAPE = data.load_real()
    EVENTS = st.basket_events(TAPE)
else:
    TAPE = EVENTS = None
print("real cache present:", HAVE_REAL, "| tickers:",
      0 if TAPE is None else len(TAPE), "| confirmed divergences:",
      0 if EVENTS is None else len(EVENTS))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a fading RSI really call the bottom? 📉📈\n"
            "### Bullish RSI divergence — one of the most-taught chart patterns, tested "
            "against 16 years of daily data\n\n"
            + BADGES +
            "Open any technical-analysis course and you'll meet this pattern in the first "
            "week: price drops to a new low, but the RSI — a bounded momentum gauge — makes a "
            "*higher* low at the same point. \"Momentum is diverging from price,\" the story "
            "goes. \"The sellers are running out of gas. Buy the divergence.\"\n\n"
            "It's a clean, visual, satisfying pattern to spot on a chart. The question is "
            "whether it actually predicts anything.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** We detect the pattern algorithmically — no eyeballing charts "
            "— on SPY plus a five-name liquid basket (QQQ, IWM, AAPL, MSFT, NVDA), 2010→2026. "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the pattern predict a bounce? | **No.** Across **{R['n_events']}** "
            "algorithmically confirmed divergences on six liquid tickers, the pattern's "
            "forward returns are **worse**, not better, than an ordinary day on the same "
            "tickers — at every horizon we tested (5, 10, 20 trading days). |\n"
            "| Does it at least beat a coin flip? | **No — it doesn't even beat a random "
            "date.** A random signal of the exact same size, on the exact same tickers, beats "
            f"the real pattern on **61-83%** of tries. |\n"
            "| Is that because the market was flat? | **No — the opposite.** This basket "
            f"had a strong bull-market drift (an unconditional day already wins **"
            f"{R['uncond_hit10']:.0f}%** of the time at 10 days); the divergence signal's hit "
            "rate is *below* that bar, not above it. |\n"
            "| Can costs save it? | **There's nothing to save.** The edge is already negative "
            "or flat before a single basis point of cost is charged. |\n\n"
            "> The picture on the chart is real. The edge behind it isn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When price makes a new low but RSI(14) doesn't, the down-move has lost its "
            "momentum. Sellers are exhausted; a reversal is imminent. Enter long on "
            "confirmation.\"*\n\n"
            "It's taught in every technical-analysis textbook, and it *feels* mechanical — "
            "you can point at exactly two spots on a chart and draw two trendlines that "
            "visibly slope in opposite directions. That visual cleanliness is a big part of "
            "why it's so widely believed."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is a genuinely useful timing tool: a rule-based way to catch "
            "reversals *before* price confirms them with new highs, using nothing but two "
            "indicators everyone already has on their screen. It would also validate the "
            "broader idea that momentum oscillators carry forward-looking information beyond "
            "what price itself already tells you.\n\n"
            "So we ask: does the pattern actually predict a bounce, and — crucially — does it "
            "beat the *bar this basket sets on its own* (a strong bull-market drift), not just "
            "an arbitrary 50%?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **Find every swing low, honestly.** A swing low can only be *known* 5 trading "
            "days after it prints — you need to see the following days to know nothing "
            "undercut it. That's how the pattern is defined, not a look-ahead shortcut.\n"
            "- **Flag the divergence.** Compare each confirmed swing low to the previous one: "
            "lower price, higher RSI(14) = bullish divergence.\n"
            "- **Trade it fairly.** Enter at the next session's open (zero look-ahead), hold "
            "5/10/20 trading days, exit at the close.\n"
            "- **Compare against the RIGHT bar.** Not 50% — an unconditional day on the same "
            "tickers, *and* a random signal of the identical size, since this basket already "
            "trends up on its own."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            f"**{R['n_events']} confirmed bullish divergences** turned up across the six "
            "tickers over 16+ years — not rare, not overfit to one stock. Here's how they did "
            "against an ordinary day on the same basket:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    hs = {h: st.headline_stats(TAPE, EVENTS, h) for h in (5, 10, 20)}\n"
            "    sig = [hs[h]['sig_mean_bps'] for h in (5, 10, 20)]\n"
            "    base = [hs[h]['base_mean_bps'] for h in (5, 10, 20)]\n"
            "else:\n"
            "    sig = [R['headline'][h][0] for h in (5, 10, 20)]\n"
            "    base = [R['headline'][h][1] for h in (5, 10, 20)]\n"
            "x = np.arange(3); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.bar(x - w/2, sig, w, color=RED, label='after a divergence signal')\n"
            "ax.bar(x + w/2, base, w, color=GREY, label='an ordinary day (unconditional)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['5d', '10d', '20d'])\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward return (bps)')\n"
            "ax.set_title('The pattern earns LESS than an ordinary day, at every horizon')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('signal:', [round(v,1) for v in sig], ' unconditional:', [round(v,1) for v in base])"
        ),
        md(
            "That's the whole headline right there: the red bars (after a divergence signal) "
            "sit **below** the grey bars (an ordinary day) at every horizon. If the pattern "
            "carried real information, red should be taller than grey — instead it's smaller, "
            "every single time.\n\n"
            "**Maybe it's just noisy — how does it do against pure luck?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl10 = st.random_signal_placebo(TAPE, EVENTS, 10, n_draws_per_seed=100, n_seeds=6)\n"
            "    obs, draws = pl10['obs']*1e4, pl10['draws']*1e4\n"
            "else:\n"
            "    obs = R['placebo'][10][0]\n"
            "    rng = np.random.default_rng(669)\n"
            "    draws = rng.normal(R['placebo'][10][1], R['placebo'][10][2], 2400)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85,\n"
            "        label='random signal, same count, same tickers (light in-notebook run)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'the REAL divergence signal ({obs:+.0f} bps)')\n"
            "ax.set_xlabel('mean 10-day forward return of a random-signal draw (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"canonical result: random beats real on {R['placebo'][10][3]*100:.0f}% of draws\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"real signal {obs:+.1f} bps vs canonical placebo mean \"\n"
            "      f\"{R['placebo'][10][1]:+.1f} bps -> p(random >= real) = {R['placebo'][10][3]:.3f}\")"
        ),
        md(
            f"The red line — the real pattern — sits **inside** the cloud of random "
            f"draws, not off to the right of it. At the 10-day horizon, a random signal of "
            f"the identical size beats the real thing on **{R['placebo'][10][3]*100:.0f}%** "
            "of tries. That's the cleanest possible verdict: this isn't a weak signal, it's "
            "statistically indistinguishable from — and slightly worse than — chance.\n\n"
            "**One more honest check: is the basket just flat, so a coin flip looks good by "
            "default?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fwd10 = st.basket_forward_series(TAPE, 10)\n"
            "    allv = np.concatenate([s.dropna().values for s in fwd10.values()])\n"
            "    uncond_hit = float((allv > 0).mean()) * 100\n"
            "    sig_hit = hs[10]['hit_rate'] * 100\n"
            "else:\n"
            "    uncond_hit, sig_hit = R['uncond_hit10'], R['headline'][10][6]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['divergence\\nsignal', 'unconditional\\n(any day)'], [sig_hit, uncond_hit],\n"
            "       color=[RED, GREY], width=.55)\n"
            "ax.axhline(50, ls='--', c='k', lw=1, label='a fair coin (50%)')\n"
            "for i,v in enumerate([sig_hit, uncond_hit]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('hit rate at 10 trading days (%)'); ax.set_ylim(0, 75)\n"
            "ax.set_title('The bull-market drift beats a coin on its own — divergence trails it')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'signal hit rate {sig_hit:.1f}%  vs unconditional {uncond_hit:.1f}%')"
        ),
        md(
            f"Both bars clear 50% — this basket simply went up a lot from 2010 to 2026. But "
            f"the divergence signal ({R['headline'][10][6]:.1f}%) sits **below** the "
            f"unconditional bar ({R['uncond_hit10']:.1f}%), not above it. \"Beats a coin\" was "
            "never the right question — \"beats doing nothing\" is, and the pattern fails "
            "that one too."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Divergence-flagged trades trail the unconditional baseline "
            "at every horizon (5, 10, 20 days); a random signal of the same size beats the "
            "real pattern most of the time.\n"
            "- **Tradability — Mirage.** There's no gross edge to protect from costs in the "
            "first place, and one trade lost **−14.3%** on its own.\n"
            "- **\"Beats a random signal?\" — Busted.** The fairest control available — same "
            "tickers, same count, random timing — outperforms the textbook pattern."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The visual trap.** RSI divergence is easy to *see* on a chart because you're "
            "choosing which two swing lows to connect after the fact. Algorithmic, "
            "non-cherry-picked detection removes that freedom — and the edge disappears "
            "with it.\n"
            "- **Sibling studies:** [109-obv-divergence](../../109-obv-divergence/) runs the "
            "same divergence *shape* on volume instead of RSI — also a Mirage. "
            "[75-knee-jerk](../../75-knee-jerk/) tests plain RSI(2) mean reversion (no "
            "divergence at all) and finds something real — worth reading side by side with "
            "this study to see which momentum ideas hold up and which don't.\n\n"
            "*Think a tighter swing-detection window or a different oscillator would "
            "resurrect the pattern? Show a net, certifiable edge against the random-signal "
            "control — not just against 50% — and we'll take another look.*"
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
            "# RSI-Divergence — a quantitative teardown 🔬\n"
            "### Confirmed swing-low pairing · Welch/HAC splits · a 20-seed random-signal "
            "placebo · the retail-era contrast · a cost-timer · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **a lower price low with a higher RSI(14) low marks a reversal** — "
            "is a textbook pattern with no stated economic mechanism beyond crowd psychology. "
            "The job here is to detect it algorithmically (never by eye), measure it honestly, "
            "and ask the only question that pays: *does it beat the basket's own drift?*\n\n"
            "> ⚠️ **Data note.** Daily OHLC, SPY + QQQ/IWM/AAPL/MSFT/NVDA, 2010-01-04 → "
            "2026-06-30, yfinance, cached. No hardcoded calendar — the pattern is purely "
            "algorithmic. No survivorship (index ETFs + still-listed mega-caps), though the "
            "basket does skew bullish (named explicitly below). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints " +
            ", ".join(f"`{v}`" for v in R["fp"].values()) + ").\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | gap vs unconditional **{R['headline'][10][2]:+.1f} bps** "
            f"(10d), Welch **t = {R['headline'][10][3]:.2f}**, NW "
            f"**t = {R['headline'][10][4]:.2f}**; random-signal placebo beats it on "
            f"**{R['placebo'][10][3]*100:.0f}%** of draws |\n"
            f"| **Tradability** | `MIRAGE` | net {R['tw_net5']:+.1f} bps/trade at 5 bps cost, "
            f"but gross is already at/below the unconditional bar; worst trade "
            f"{R['tw_worst']:.0f} bps |\n"
            f"| **Beats a random signal?** | `BUSTED` | p(random ≥ observed) = "
            f"{R['placebo'][5][3]:.2f} / {R['placebo'][10][3]:.2f} / {R['placebo'][20][3]:.2f} "
            "at 5/10/20d — random wins the majority of the time at every horizon |\n\n"
            "> 💡 In plain words: the pattern is not a weak edge fighting costs — it never "
            "clears the starting line."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $D_t \\in \\{0,1\\}$ flag a **confirmed** bullish divergence: a swing low at "
            "$t$ with price $P_t < P_{t'}$ and RSI$_t$ > RSI$_{t'}$ vs the previous confirmed "
            "swing low $t'$. A swing low is only knowable `order`=5 bars after it prints (the "
            "definition itself requires future bars — not a look-ahead violation any more than "
            "RSI(14) needing 14 days of history). The claims:\n\n"
            "- **H₁ (signal).** $E[r_{t+1:t+h} \\mid D_t=1] > E[r_{t+1:t+h}]$ (unconditional) "
            "for $h \\in \\{5,10,20\\}$.\n"
            "- **H₂ (beats random).** The divergence signal beats a random signal of the same "
            "count, same tickers.\n"
            "- **H₃ (bankable).** Net of costs, long-only, the pattern pays.\n\n"
            "We find **H₁ rejected** (negative gap at all three horizons), **H₂ rejected** "
            "(random beats real 61-83% of the time) and **H₃ moot** — there's no gross edge to "
            "protect from costs."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Signal-vs-unconditional uses a **Welch t** on the group split. Because h-day "
            "forward-return windows overlap (two divergences within h days of each other on "
            "the same or different tickers share tape), we cross-check with a pooled "
            "**Newey-West (HAC, lag=h)** dummy regression. The hit rate carries a **Wilson "
            "interval**. The random-signal placebo draws the observed signal COUNT per ticker "
            "from random eligible dates, **20 seeds × 200 draws**, and reports a right-tail "
            "p-value (share of random draws whose mean ≥ the observed divergence mean — the "
            "claim predicts a positive edge, so a high p means random usually does *better*)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY + QQQ/IWM/AAPL/MSFT/NVDA, {R['start']} → {R['end']}, daily "
            "OHLC (yfinance). No survivorship (index ETFs + still-listed mega-caps).\n"
            "- **Pattern.** Confirmed 11-bar-fractal swing lows (`order=5`); bullish "
            "divergence = lower price low, higher RSI(14) low vs the previous confirmed swing.\n"
            f"- **Events.** {R['n_events']} confirmed divergences across 6 tickers "
            f"({', '.join(f'{k} {v}' for k, v in R['by_ticker'].items())}).\n"
            "- **Execution (single documented lag).** Enter next session's open after "
            "confirmation (zero look-ahead), exit at the close h sessions later.\n"
            "- **Comparisons.** (a) unconditional forward-return distribution, same formula, "
            "every day; (b) random-signal placebo, matched count, 20 seeds × 200 draws.\n"
            "- **Cross-check.** Newey-West (lag=h) on the pooled dummy regression.\n"
            "- **Sub-period.** Split 2019-10-01 (zero-commission retail era), tested as a "
            "*difference*, not eyeballed.\n"
            "- **Third axis.** Long-only cost-timer, 2 × one-way × NAV, 5/10 bps.\n"
            "- **Control.** Synthetic random walk, planted bounce injected on the pipeline's "
            "own flagged dates; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split, three horizons\n\n"
            "Welch t on the divergence-vs-unconditional gap, with the NW(lag=h) cross-check "
            "for the overlap in forward-return windows."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hs = {h: st.headline_stats(TAPE, EVENTS, h) for h in (5, 10, 20)}\n"
            "    for h in (5, 10, 20):\n"
            "        s = hs[h]\n"
            "        print(f\"h={h:>2d}d: signal {s['sig_mean_bps']:+7.2f} bps (n={s['n_sig']})  \"\n"
            "              f\"vs unconditional {s['base_mean_bps']:+7.2f} bps (n={s['n_base']:,})  \"\n"
            "              f\"gap {s['gap_bps']:+7.2f} bps  Welch t={s['welch_t']:+.2f}  \"\n"
            "              f\"NW t={s['nw_t']:+.2f}  hit {s['hit_rate']*100:.1f}%\")\n"
            "    gaps = [hs[h]['gap_bps'] for h in (5, 10, 20)]\n"
            "    ts = [hs[h]['welch_t'] for h in (5, 10, 20)]\n"
            "else:\n"
            "    gaps = [R['headline'][h][2] for h in (5, 10, 20)]\n"
            "    ts = [R['headline'][h][3] for h in (5, 10, 20)]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['5d', '10d', '20d'], gaps, color=RED, width=.55)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('gap vs unconditional (bps)')\n"
            "a1.set_title('Signal - unconditional: negative at every horizon')\n"
            "a2.bar(['5d', '10d', '20d'], ts, color=[RED if abs(t)>=2 else GREY for t in ts], width=.55)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('Welch t'); a2.set_title('Nowhere near the ±2 bar')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the gap is negative at 5d ({R['headline'][5][2]:+.1f} bps), "
            f"10d ({R['headline'][10][2]:+.1f} bps) and 20d ({R['headline'][20][2]:+.1f} bps) "
            "— the divergence signal is a laggard against the same basket's ordinary day, and "
            "no *t* gets within 1.2 of the ±2 bar in either direction."
        ),
        md(
            "### 4b · The random-signal placebo — the fairest bar\n\n"
            "20 seeds × 200 draws of the exact signal count per ticker, from random eligible "
            "dates: p = share of random draws with mean ≥ the observed divergence mean."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = {h: st.random_signal_placebo(TAPE, EVENTS, h, n_draws_per_seed=100, n_seeds=6)\n"
            "          for h in (5, 10, 20)}\n"
            "    obs = [pl[h]['obs']*1e4 for h in (5, 10, 20)]\n"
            "    pm = [pl[h]['placebo_mean']*1e4 for h in (5, 10, 20)]\n"
            "    ps = [pl[h]['placebo_sd']*1e4 for h in (5, 10, 20)]\n"
            "else:\n"
            "    obs = [R['placebo'][h][0] for h in (5, 10, 20)]\n"
            "    pm = [R['placebo'][h][1] for h in (5, 10, 20)]\n"
            "    ps = [R['placebo'][h][2] for h in (5, 10, 20)]\n"
            "x = np.arange(3); w = .35\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(x - w/2, obs, w, color=RED, label='real divergence signal')\n"
            "ax.bar(x + w/2, pm, w, yerr=ps, color=GREY, capsize=4, label='random-signal placebo (±1 sd)')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['5d','10d','20d'])\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean forward return (bps)')\n"
            "ax.set_title('The real pattern sits inside — or below — the random cloud')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('canonical p-values (results.md):',\n"
            "      {h: R['placebo'][h][3] for h in (5, 10, 20)})"
        ),
        md(
            f"> 💡 In plain words: the canonical (20-seed × 200-draw) p-values are "
            f"**{R['placebo'][5][3]:.3f} / {R['placebo'][10][3]:.3f} / {R['placebo'][20][3]:.3f}** "
            "at 5/10/20 days — a random signal beats the real pattern on the clear majority of "
            "draws at every horizon. This is the single most decisive number in the study: it "
            "isn't that the divergence signal is *weak*, it's that doing nothing in particular "
            "(same tickers, same count, random timing) usually wins."
        ),
        md(
            "### 4c · Era contrast — pre vs post the zero-commission retail era\n\n"
            "Split at **2019-10-01** (Schwab/Robinhood cut equity commissions to zero — retail "
            "chart-pattern trading, of which RSI divergence is a textbook example, went "
            "frictionless and crowded), tested as a difference, not eyeballed."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.era_contrast(TAPE, EVENTS, 10, R['era_split'])\n"
            "    e, l, n_e, n_l, dt = (ec['early_bps'], ec['late_bps'], ec['n_early'],\n"
            "                          ec['n_late'], ec['welch_t_diff'])\n"
            "else:\n"
            "    e, l, n_e, n_l, dt = (R['era_early'], R['era_late'], R['era_early_n'],\n"
            "                          R['era_late_n'], R['era_diff_t'])\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar([f'2010 - 2019-10\\n(n={n_e})', f'2019-10 - 2026\\n(n={n_l})'], [e, l],\n"
            "       color=[GREY, AMBER], width=.5)\n"
            "for i,v in enumerate([e, l]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('signal mean, h=10d (bps)')\n"
            "ax.set_title(f'No certified decay or crowding story (diff t = {dt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early {e:+.1f} bps (n={n_e})  late {l:+.1f} bps (n={n_l})  diff t = {dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the point estimate is larger post-2019 "
            f"({R['era_early']:+.1f} → {R['era_late']:+.1f} bps), but the difference is "
            f"**not certified** (t = {R['era_diff_t']:+.2f}) — with only "
            f"{R['era_early_n']} and {R['era_late_n']} events per era there's no basis to "
            "claim decay or a crowding effect. Both eras are already statistically "
            "indistinguishable from noise on their own."
        ),
        md(
            "### 4d · The third axis — a timer with costs\n\n"
            "Long-only (this is a bullish pattern), enter next open, exit at the close 10 "
            "sessions later, 2 × one-way costs per round trip."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = {cb: st.timer_with_costs(TAPE, EVENTS, 10, cost_bps=cb) for cb in (5.0, 10.0)}\n"
            "    g = rows[5.0]['gross_bps']; n5, n10 = rows[5.0]['net_bps'], rows[10.0]['net_bps']\n"
            "    hit, worst, best = rows[5.0]['hit_rate']*100, rows[5.0]['worst_bps'], rows[5.0]['best_bps']\n"
            "else:\n"
            "    g, n5, n10 = R['tw_gross'], R['tw_net5'], R['tw_net10']\n"
            "    hit, worst, best = R['tw_hit'], R['tw_worst'], R['tw_best']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['gross', 'net 5 bps', 'net 10 bps'], [g, n5, n10], color=[GREY, AMBER, AMBER], width=.6)\n"
            "for i,v in enumerate([g, n5, n10]): ax.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('bps per trade')\n"
            "ax.set_title(f'Never had a gross edge to protect (hit {hit:.1f}%, worst {worst:.0f} bps)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.1f} -> net {n5:+.1f} / {n10:+.1f} bps;  hit {hit:.1f}%;  '\n"
            "      f'worst {worst:+.1f} bps;  best {best:+.1f} bps')"
        ),
        md(
            f"> 💡 In plain words: gross **{R['tw_gross']:+.1f} bps/trade** is already at or "
            f"below the unconditional bar ({R['uncond_hit10']:.1f}% hit rate vs "
            f"{R['headline'][10][6]:.1f}% for the signal); 5-10 bps of cost turns a marginal "
            f"gross number into {R['tw_net5']:+.1f} / {R['tw_net10']:+.1f} bps net — and the "
            f"worst single trade ({R['tw_worst']:.0f} bps) is larger in magnitude than 25 "
            "average wins stacked together. There was never a gross edge to protect."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic random walk with engineered periodic dips (so genuine swing lows "
            "recur); a TUNABLE bounce is planted exactly on the dates the study's **own "
            "detector** flags — never a hand-tuned proxy. The null (bounce=0) is checked over "
            "**20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    world = data.synthetic_world(bounce=0.0, seed=669 + s_)\n"
            "    null_ts.append(st.synthetic_detect(world, h=10)['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "world = data.synthetic_world(bounce=0.02, seed=669)\n"
            "planted = st.synthetic_detect(world, h=10)\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (bounce=0), 20 seeds')\n"
            "ax.scatter([1], [planted['welch_t']], color=RED, s=90, zorder=5,\n"
            "           label='planted bounce = +2.0% over 5 bars')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (signal vs rest)')\n"
            "ax.set_title('Control: no null fires; a planted bounce lights up hard')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = '\n"
            "      f\"{planted['welch_t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses "
            f"the bar; planting a 2% bounce exactly where the detector's own pipeline flags a "
            f"signal reads t = {R['syn_planted_t']:.2f} (NW t = {R['syn_planted_nw']:.2f}). The "
            "machinery is unbiased and has power — the real-tape null result is the genuine "
            "article, not a blind spot. *(A faithful-engine / power check only — never cited "
            "in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — gap vs unconditional negative at every horizon "
            f"({R['headline'][5][2]:+.1f} / {R['headline'][10][2]:+.1f} / "
            f"{R['headline'][20][2]:+.1f} bps at 5/10/20d), Welch t "
            f"{R['headline'][5][3]:+.2f} / {R['headline'][10][3]:+.2f} / "
            f"{R['headline'][20][3]:+.2f} (NW cross-check matches); a random signal of the "
            f"same size beats the real pattern on "
            f"{R['placebo'][5][3]*100:.0f}% / {R['placebo'][10][3]*100:.0f}% / "
            f"{R['placebo'][20][3]*100:.0f}% of draws.\n"
            f"- **Tradability `MIRAGE`** — no gross edge survives contact with the "
            "unconditional bar; costs only make a losing proposition worse; worst single "
            f"trade {R['tw_worst']:.0f} bps.\n"
            "- **\"Beats a random signal?\" `BUSTED`** — the fairest control on this desk "
            "outperforms the textbook pattern at every horizon tested. The visual cleanliness "
            "of a divergence on a chart does not survive algorithmic, non-cherry-picked "
            "detection."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Why the visual pattern feels so much stronger than the data.** Chart-reading "
            "involves choosing *which* two swing lows to connect after you've already seen "
            "what happened next — the algorithmic detector has no such freedom, and the edge "
            "evaporates with the hindsight.\n"
            "- **The general lesson.** Divergence claims recur across the technical-analysis "
            "canon (price/volume, price/RSI, price/MACD…) — the shared structure (two "
            "confirmed extrema disagreeing in direction) is worth testing systematically "
            "rather than indicator-by-indicator; this desk's [109-obv-divergence]"
            "(../../109-obv-divergence/) does the volume version and finds the same nothing.\n"
            "- **Dedup map:** [109-obv-divergence](../../109-obv-divergence/) (volume "
            "divergence), [75-knee-jerk](../../75-knee-jerk/) (RSI(2) mean reversion, no "
            "divergence), [301-triple-rsi](../../301-triple-rsi/) (multi-timeframe RSI "
            "alignment), [428-stochastic-rsi](../../428-stochastic-rsi/) (Stochastic-on-RSI "
            "timer), [178-cci](../../178-cci/) (a different oscillator's breach rule) — none "
            "test the confirmed-swing-low pairing this study does.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
